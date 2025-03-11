import os
import csv
import argparse


def format_number(value):
    """Formats a number to have exactly four decimal places."""
    try:
        return "{:.4f}".format(float(value))
    except ValueError:
        return value  # Return original if it's not a number


def process_csv(file_path):
    """Reads a CSV file, updates numeric values, and rewrites the file."""
    with open(file_path, 'r', newline='') as infile:
        reader = csv.reader(infile)
        rows = [
            [format_number(cell) for cell in row]
            for row in reader
        ]

    with open(file_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)
    print(f"Processed: {file_path}")


def scan_directory(root_dir, ignored_folders):
    """Recursively scans directories for CSV files, skipping ignored folders."""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Remove ignored folders from the search
        dirnames[:] = [d for d in dirnames if d not in ignored_folders]

        for filename in filenames:
            if filename.endswith(".csv"):
                process_csv(os.path.join(dirpath, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix CSV decimal places in all subdirectories.")
    parser.add_argument("directory", help="Root directory to scan.")
    parser.add_argument("--ignore", nargs='*', default=[], help="Folder names to ignore.")
    args = parser.parse_args()

    scan_directory(args.directory, args.ignore)
