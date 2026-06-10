import sys
from pipeline import process_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <file1> [file2] ...")
        sys.exit(1)

    for path in sys.argv[1:]:
        print(f"Processing: {path}")
        process_file(path)