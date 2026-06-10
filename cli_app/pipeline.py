import asyncio
import gc
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from itext2kg.documents_distiller import DocumentsDistiller
from itext2kg import iText2KG_Star

from config import get_llm, get_embeddings, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


# ── Schema ────────────────────────────────────────────────────────────────────

class SimpleTriple(BaseModel):
    subject:   str = Field(default="", description="Субјект - максимум 3 збора на македонски")
    predicate: str = Field(default="", description="Предикат - еден глагол на македонски, максимум 3 збора")
    obj:       str = Field(default="", description="Објект - максимум 3 збора на македонски")

class ChunkTriples(BaseModel):
    triples: List[SimpleTriple] = Field(default_factory=list, description="Листа од максимум 5 тројки")

    @field_validator("triples", mode="before")
    @classmethod
    def limit_triples(cls, v):
        return v[:5] if isinstance(v, list) else []


IE_QUERY = """
# ПРАВИЛА - СТРОГО СЛЕДИ:
- Извлечи МАКСИМУМ 5 тројки (субјект, предикат, објект) од текстот.
- СЕ МОРА да биде на МАКЕДОНСКИ ЈАЗИК.
- Субјект и Објект: максимум 3 збора, конкретен поим или ентитет.
- Предикат: еден глагол, максимум 3 збора (пр: "предизвикува", "се состои од", "се наоѓа во").
- НИКОГАШ не повторувај иста тројка.
- НИКОГАШ не измислувај факти кои не се во текстот.
- Ако нема јасни односи, врати празна листа.
"""


# ── Triple cleaning & validation ──────────────────────────────────────────────

_TRIVIAL   = {"yes", "no", "none", "unknown", "да", "не", "нема", "непознато", "непозната", "непознат"}
_BAD_WORDS = {"тоа", "ова", "овој", "оваа", "тие", "тој", "таа", "нешто", "некој", "сите", "никој", "ништо"}
_PREDICATE_MAP = {
    "предизвикува": "causes", "предизвикуваат": "causes",
    "содржи": "contains",     "состои": "consists_of",
    "припаѓа": "belongs_to",  "наоѓа": "located_in",
    "вклучува": "includes",   "опфаќа": "includes",
    "дефинира": "defines",    "претставува": "represents",
    "настанува": "formed_by", "формира": "forms",
    "граничи": "borders",     "влијае": "affects",
    "изучува": "studies",
}


def clean_entity(e: str) -> str:
    if not e:
        return ""
    e = re.sub(r"\s+", " ", str(e).strip())
    e = re.sub(r"[,;:.!?]+$", "", e).strip("()[]{} ")
    if not e or e.lower() in _TRIVIAL or len(e.split()) > 5:
        return ""
    return e


def clean_predicate(p: str) -> str:
    if not p:
        return ""
    p = str(p).lower().strip()
    p = re.sub(r"[^\w\sа-шѓќљњѕџјА-ШЃЌЉЊЅЏЈ]", "", p)
    p = re.sub(r"\s+", "_", p)
    if len(p) > 40 or len(p) < 2:
        return ""
    return _PREDICATE_MAP.get(p, p)


def is_valid_triple(s: str, p: str, o: str) -> bool:
    if not s or not p or not o:
        return False
    if len(s) < 2 or len(o) < 2 or len(p) < 2:
        return False
    if s.lower() == o.lower():
        return False
    if s.lower() in _BAD_WORDS or o.lower() in _BAD_WORDS:
        return False
    if len(s) > 60 or len(o) > 60 or len(p) > 60:
        return False
    # reject purely English entity pairs
    eng = re.compile(r"^[a-zA-Z\s]+$")
    if eng.match(s) and eng.match(o):
        return False
    return True


def safe_get_name(obj, default="unknown") -> str:
    if obj is None:
        return default
    if hasattr(obj, "name") and obj.name:
        return str(obj.name)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict) and "name" in obj:
        return str(obj["name"])
    return default


def deduplicate_triples(triples: List[tuple]) -> List[tuple]:
    seen, unique = set(), []
    for s, p, o in triples:
        key = (s.lower(), p.lower(), o.lower())
        if key not in seen:
            seen.add(key)
            unique.append((s, p, o))
    return unique


# ── File reader ───────────────────────────────────────────────────────────────

def read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if p.suffix == ".txt":
        return p.read_text(encoding="utf-8")

    elif p.suffix == ".pdf":
        import fitz  # PyMuPDF — better Macedonian text extraction than pypdf
        text = ""
        with fitz.open(str(p)) as doc:
            for page in doc:
                text += f"\n\n[PAGE {page.number + 1}]\n{page.get_text()}"
        return text

    elif p.suffix == ".docx":
        from docx import Document
        return "\n".join(para.text for para in Document(str(p)).paragraphs)

    else:
        raise ValueError(f"Unsupported file type: {p.suffix}  (supported: .txt, .pdf, .docx)")


# ── Chunker ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    sentences = re.split(r"(?<=[.!?؟])\s+|\n+", text)
    chunks, current = [], ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current + sent) <= chunk_size:
            current += sent + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + sent + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ── Log writer ────────────────────────────────────────────────────────────────

def save_triples_log(file_path: str, triples: List[tuple]):
    source    = Path(file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = source.parent / "logs" / f"{source.stem}_kg_log_{timestamp}.txt"

    lines = [
        "=" * 60,
        "KNOWLEDGE GRAPH LOG",
        f"Source file : {file_path}",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        f"\nTRIPLES ({len(triples)})",
        "-" * 40,
    ]
    for i, (s, p, o) in enumerate(triples, 1):
        lines.append(f"  {i:>3}. ({s}) --[{p}]--> ({o})")
    lines.append("\n" + "=" * 60)

    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  📄 Log saved: {log_path}")
    return log_path


# ── Retry wrapper ─────────────────────────────────────────────────────────────

class PipelineError(Exception):
    pass


async def distill_with_retry(chunk: str, distiller, max_retries: int = 3):
    cleaned = chunk.replace("{", "[").replace("}", "]")
    for attempt in range(max_retries):
        try:
            return await distiller.distill(
                documents=[cleaned],
                IE_query=IE_QUERY,
                output_data_structure=ChunkTriples,
            )
        except asyncio.TimeoutError:
            print(f"    ⚠ Attempt {attempt + 1} timed out")
        except Exception as e:
            msg = str(e)[:100]
            print(f"    ⚠ Attempt {attempt + 1} failed: {type(e).__name__}: {msg}")
            if attempt == max_retries - 1:
                raise PipelineError(f"Distillation failed after {max_retries} attempts: {msg}")
        await asyncio.sleep(2 ** attempt)


# ── Core pipeline ─────────────────────────────────────────────────────────────

async def _run(file_path: str):
    llm        = get_llm()
    embeddings = get_embeddings()
    start_time = time.time()

    # ── Phase 1: Read & chunk ─────────────────────────────────────────────────
    print("  [1/3] Reading and chunking document...")
    text   = read_file(file_path)
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    print(f"        {len(text)} chars → {len(chunks)} chunks")

    # ── Phase 2: Distill each chunk → triples + semantic blocks ──────────────
    print(f"  [2/3] Distilling {len(chunks)} chunks...")
    distiller       = DocumentsDistiller(llm_model=llm)
    all_triples     = []
    semantic_blocks = []

    for i, chunk in enumerate(chunks, 1):
        try:
            result = await distill_with_retry(chunk, distiller)
            if result and hasattr(result, "triples") and result.triples:
                for triple in result.triples:
                    s = clean_entity(triple.subject)
                    p = clean_predicate(triple.predicate)
                    o = clean_entity(triple.obj)
                    if is_valid_triple(s, p, o):
                        all_triples.append((s, p, o))
                        semantic_blocks.append(f"{s} {p} {o}")
                print(f"        [{i}/{len(chunks)}] {len(result.triples)} triples")
            else:
                semantic_blocks.append(chunk[:400])
        except PipelineError as e:
            print(f"        [{i}/{len(chunks)}] ⚠ fallback — {str(e)[:60]}")
            semantic_blocks.append(chunk[:400])
        gc.collect()

    # deduplicate blocks before STAR
    seen_blocks, unique_blocks = set(), []
    for b in semantic_blocks:
        key = b.lower().strip()
        if key not in seen_blocks and len(key) > 10:
            seen_blocks.add(key)
            unique_blocks.append(b)
    semantic_blocks = unique_blocks
    print(f"        Distiller triples: {len(all_triples)} | Unique blocks for STAR: {len(semantic_blocks)}")

    # ── Phase 3: STAR on semantic blocks ─────────────────────────────────────
    print(f"  [3/3] Running STAR on {len(semantic_blocks)} blocks...")
    star_count = 0
    star_model = iText2KG_Star(llm_model=llm, embeddings_model=embeddings)
    today      = datetime.now().strftime("%Y-%m-%d")

    for i, block in enumerate(semantic_blocks, 1):
        try:
            kg = await star_model.build_graph(
                sections=[block],
                ent_threshold=0.5,
                rel_threshold=0.4,
                observation_date=today,
            )
            if hasattr(kg, "relationships"):
                for rel in kg.relationships:
                    s = clean_entity(safe_get_name(getattr(rel, "startEntity", None)))
                    o = clean_entity(safe_get_name(getattr(rel, "endEntity",   None)))
                    p = clean_predicate(safe_get_name(getattr(rel, "name", None), ""))
                    if is_valid_triple(s, p, o):
                        all_triples.append((s, p, o))
                        star_count += 1
        except Exception as e:
            print(f"        Block {i} failed: {str(e)[:80]}")

    print(f"        STAR triples: {star_count}")

    # ── Post-processing ───────────────────────────────────────────────────────
    final = deduplicate_triples([
        (clean_entity(s), clean_predicate(p), clean_entity(o))
        for s, p, o in all_triples
        if is_valid_triple(clean_entity(s), clean_predicate(p), clean_entity(o))
    ])

    elapsed = (time.time() - start_time) / 60
    print(f"\n  ✅ Done in {elapsed:.1f}min — {len(final)} unique triples")

    if final:
        print("  Sample:")
        for s, p, o in final[:5]:
            print(f"    ({s}) --[{p}]--> ({o})")

    save_triples_log(file_path, final)
    return final


# ── Public entry point (called from main.py) ──────────────────────────────────

def process_file(file_path: str):
    asyncio.run(_run(file_path))