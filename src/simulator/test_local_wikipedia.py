"""
Test script for local Wikipedia functionality.

Usage:
    python test_local_wikipedia.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.local_wikipedia import LocalWikipedia


def test_basic_lookup():
    """Test basic Wikipedia lookup."""
    print("=" * 60)
    print("Test 1: Basic Lookup")
    print("=" * 60)

    wiki = LocalWikipedia(
        wiki_dir="../../data/wikipedia/extracted",
        cache_size=100
    )

    # Test exact match
    terms = ["elephant", "python", "machine learning", "computer", "image"]

    for term in terms:
        summary = wiki.get_summary(term, sentences=2)
        if summary:
            print(f"\n✓ {term}:")
            print(f"  {summary[:150]}...")
        else:
            print(f"\n✗ {term}: Not found")

    print()


def test_fuzzy_matching():
    """Test fuzzy matching."""
    print("=" * 60)
    print("Test 2: Fuzzy Matching")
    print("=" * 60)

    wiki = LocalWikipedia(wiki_dir="../../data/wikipedia/extracted")

    # Test variations
    test_cases = [
        ("elephants", "elephant"),  # Plural
        ("The elephant", "elephant"),  # With article
        ("Python", "python"),  # Capitalization
    ]

    for query, expected in test_cases:
        summary = wiki.get_summary(query, sentences=1)
        if summary:
            print(f"\n✓ '{query}' → Found")
            print(f"  {summary[:100]}...")
        else:
            print(f"\n✗ '{query}' → Not found")

    print()


def test_search():
    """Test search functionality."""
    print("=" * 60)
    print("Test 3: Search")
    print("=" * 60)

    wiki = LocalWikipedia(wiki_dir="../../data/wikipedia/extracted")

    queries = ["neural", "vision", "language"]

    for query in queries:
        print(f"\nSearching for '{query}':")
        results = wiki.search(query, top_k=3)

        if results:
            for i, (title, summary) in enumerate(results, 1):
                print(f"  {i}. {title}")
                print(f"     {summary[:80]}...")
        else:
            print("  No results found")

    print()


def test_performance():
    """Test performance and caching."""
    print("=" * 60)
    print("Test 4: Performance & Caching")
    print("=" * 60)

    wiki = LocalWikipedia(
        wiki_dir="../../data/wikipedia/extracted",
        cache_size=10
    )

    import time

    # First lookup (cache miss)
    start = time.time()
    summary1 = wiki.get_summary("elephant")
    time1 = time.time() - start

    # Second lookup (cache hit)
    start = time.time()
    summary2 = wiki.get_summary("elephant")
    time2 = time.time() - start

    print(f"\nFirst lookup (cache miss): {time1*1000:.2f}ms")
    print(f"Second lookup (cache hit): {time2*1000:.2f}ms")
    print(f"Speedup: {time1/time2:.1f}x")

    # Stats
    stats = wiki.get_stats()
    print(f"\nStatistics:")
    print(f"  Total articles: {stats['total_articles']:,}")
    print(f"  Cache size: {stats['cache_size']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"  Index size: {stats['index_file_size_mb']:.2f} MB")

    print()


def test_integration_example():
    """Test integration with length controller."""
    print("=" * 60)
    print("Test 5: Integration Example")
    print("=" * 60)

    wiki = LocalWikipedia(wiki_dir="../../data/wikipedia/extracted")

    # Simulate user simulator message
    message = "What do you see in the image? Can you describe the elephant?"

    # Extract nouns (simple version)
    import re
    words = re.findall(r'\b[a-z]+\b', message.lower())
    nouns = [w for w in words if len(w) > 3]  # Simple heuristic

    print(f"\nOriginal message ({len(message.split())} words):")
    print(f"  {message}")

    # Pad with Wikipedia definitions
    padded = message
    target_length = 100

    for noun in nouns:
        if len(padded.split()) >= target_length:
            break

        summary = wiki.get_summary(noun, sentences=1)
        if summary:
            addition = f" By the way, {noun} is {summary}"
            words_remaining = target_length - len(padded.split())

            if len(addition.split()) <= words_remaining:
                padded += addition
            else:
                # Truncate
                words_to_add = addition.split()[:words_remaining]
                padded += " " + " ".join(words_to_add) + "..."
                break

    print(f"\nPadded message ({len(padded.split())} words):")
    print(f"  {padded}")

    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Local Wikipedia Test Suite")
    print("=" * 60 + "\n")

    try:
        test_basic_lookup()
        test_fuzzy_matching()
        test_search()
        test_performance()
        test_integration_example()

        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run prepare_wikipedia.py first:")
        print("  python prepare_wikipedia.py --lang en --output ../../data/wikipedia")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
