"""
Script to download and prepare Wikipedia data for local use.

Usage:
    python prepare_wikipedia.py --lang en --output data/wikipedia
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_wikipedia_dump(lang: str, output_dir: Path):
    """
    Download Wikipedia dump.

    Args:
        lang: Language code (e.g., 'en', 'zh')
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dump_url = f"https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles.xml.bz2"
    output_file = output_dir / f"{lang}wiki-latest-pages-articles.xml.bz2"

    if output_file.exists():
        logger.info(f"Dump file already exists: {output_file}")
        return output_file

    logger.info(f"Downloading Wikipedia dump from {dump_url}")
    logger.info("This may take a while (file is ~20GB for English)...")

    try:
        subprocess.run([
            "wget",
            "-c",  # Continue partial downloads
            "-O", str(output_file),
            dump_url
        ], check=True)

        logger.info(f"Download complete: {output_file}")
        return output_file

    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("wget not found. Please install wget or download manually:")
        logger.error(f"  {dump_url}")
        sys.exit(1)


def extract_wikipedia_dump(dump_file: Path, output_dir: Path):
    """
    Extract Wikipedia dump to JSON format.

    Args:
        dump_file: Path to .xml.bz2 dump file
        output_dir: Output directory for extracted files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting Wikipedia dump: {dump_file}")
    logger.info(f"Output directory: {output_dir}")

    try:
        # Check if wikiextractor is installed
        subprocess.run(["python", "-m", "wikiextractor", "--help"],
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("wikiextractor not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "wikiextractor"],
                      check=True)

    # Extract with wikiextractor
    logger.info("Extracting articles (this may take 1-2 hours)...")

    try:
        subprocess.run([
            "python", "-m", "wikiextractor.WikiExtractor",
            "--json",  # Output as JSON
            "--no-templates",  # Skip templates
            "--processes", "4",  # Use 4 processes
            "--output", str(output_dir),
            str(dump_file)
        ], check=True)

        logger.info(f"Extraction complete: {output_dir}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)


def build_index(extracted_dir: Path):
    """
    Build index from extracted files.

    Args:
        extracted_dir: Directory containing extracted wiki files
    """
    logger.info("Building index...")

    # Import here to avoid circular dependency
    from local_wikipedia import LocalWikipedia

    wiki = LocalWikipedia(
        wiki_dir=str(extracted_dir),
        rebuild_index=True
    )

    stats = wiki.get_stats()
    logger.info(f"Index built successfully:")
    logger.info(f"  Total articles: {stats['total_articles']}")
    logger.info(f"  Index file: {stats['index_file']}")
    logger.info(f"  Index size: {stats['index_file_size_mb']:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Prepare Wikipedia data for local use")

    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code (default: en)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/wikipedia",
        help="Output directory (default: data/wikipedia)"
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (use existing dump)"
    )

    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction step (use existing extracted files)"
    )

    parser.add_argument(
        "--dump-file",
        type=str,
        help="Path to existing dump file (if --skip-download)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    extracted_dir = output_dir / "extracted"

    # Step 1: Download
    if not args.skip_download:
        dump_file = download_wikipedia_dump(args.lang, output_dir)
    else:
        if args.dump_file:
            dump_file = Path(args.dump_file)
        else:
            dump_file = output_dir / f"{args.lang}wiki-latest-pages-articles.xml.bz2"

        if not dump_file.exists():
            logger.error(f"Dump file not found: {dump_file}")
            sys.exit(1)

    # Step 2: Extract
    if not args.skip_extract:
        extract_wikipedia_dump(dump_file, extracted_dir)
    else:
        if not extracted_dir.exists():
            logger.error(f"Extracted directory not found: {extracted_dir}")
            sys.exit(1)

    # Step 3: Build index
    build_index(extracted_dir)

    logger.info("=" * 60)
    logger.info("Wikipedia preparation complete!")
    logger.info(f"To use in your code:")
    logger.info(f"  from local_wikipedia import LocalWikipedia")
    logger.info(f"  wiki = LocalWikipedia('{extracted_dir}')")
    logger.info(f"  summary = wiki.get_summary('elephant')")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
