"""Download and unpack the public NASA C-MAPSS turbofan dataset.

Usage:
  python scripts/download_cmapss.py --source-url <public_zip_url> --target-dir data/c-mapss

Notes:
- The script requires a direct URL to a ZIP that contains the C-MAPSS text files.
- The official landing page is https://ti.arc.nasa.gov/tech/dash/pcoe/prognostic-data-repository/#turbofan
  (fetch a direct ZIP link from there or a public mirror).
- No credentials are used; only public sources are supported.
"""

import argparse
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
import zipfile


def download_and_extract(source_url: str, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "cmapss.zip"
        print(f"Downloading {source_url} -> {zip_path}")
        urlretrieve(source_url, zip_path)
        print(f"Extracting to {target_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)

    # Flatten common nested folders and remove non-text artifacts
    for child in list(target_dir.glob("**/*.txt")):
        rel = child.relative_to(target_dir)
        flat_path = target_dir / rel.name
        if child != flat_path:
            shutil.move(str(child), flat_path)
    # Remove nested dirs if empty
    for nested in sorted(target_dir.glob("**/*"), reverse=True):
        if nested.is_dir() and not any(nested.iterdir()):
            nested.rmdir()

    print("Download complete. Files available in", target_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NASA C-MAPSS dataset")
    parser.add_argument("--source-url", required=True, help="Direct URL to a public C-MAPSS ZIP archive")
    parser.add_argument("--target-dir", default="data/c-mapss", help="Where to place the extracted files")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    download_and_extract(args.source_url, target_dir)


if __name__ == "__main__":
    main()
