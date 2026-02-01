"""
Local Wikipedia knowledge base for offline access.
Builds and uses a pre-indexed database of Wikipedia articles.
"""

import json
import os
import pickle
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class LocalWikipedia:
    """
    Local Wikipedia knowledge base using pre-processed index.

    Features:
    - Fast lookup (O(1) for exact match)
    - Fuzzy matching for partial terms
    - Caching for frequently accessed terms
    - Automatic index building from Wikipedia dumps

    Usage:
        wiki = LocalWikipedia(wiki_dir="data/wikipedia/extracted")
        summary = wiki.get_summary("elephant", sentences=2)
    """

    def __init__(self,
                 wiki_dir: str,
                 index_file: str = None,
                 cache_size: int = 1000,
                 rebuild_index: bool = False):
        """
        Initialize local Wikipedia.

        Args:
            wiki_dir: Directory containing extracted Wikipedia files
            index_file: Path to index file (default: wiki_dir/wiki_index.pkl)
            cache_size: Number of terms to cache in memory
            rebuild_index: Force rebuild of index even if exists
        """
        self.wiki_dir = Path(wiki_dir)

        if index_file is None:
            index_file = self.wiki_dir / "wiki_index.pkl"
        self.index_file = Path(index_file)

        self.cache_size = cache_size
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Load or build index
        if rebuild_index or not self.index_file.exists():
            logger.info("Building Wikipedia index...")
            self.index = self._build_index()
        else:
            logger.info(f"Loading Wikipedia index from {self.index_file}")
            self.index = self._load_index()

        logger.info(f"Wikipedia index loaded with {len(self.index)} articles")

    def _load_index(self) -> Dict[str, str]:
        """Load pre-built index from disk."""
        try:
            with open(self.index_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            logger.info("Rebuilding index...")
            return self._build_index()

    def _build_index(self) -> Dict[str, str]:
        """
        Build index from extracted Wikipedia files.

        Expected format: JSON lines with 'title' and 'text' fields.
        """
        index = {}

        if not self.wiki_dir.exists():
            logger.warning(f"Wikipedia directory not found: {self.wiki_dir}")
            logger.warning("Creating empty index. Please extract Wikipedia dump first.")
            return index

        # Find all wiki files
        wiki_files = list(self.wiki_dir.rglob("wiki_*"))

        if not wiki_files:
            logger.warning(f"No wiki files found in {self.wiki_dir}")
            return index

        logger.info(f"Processing {len(wiki_files)} wiki files...")

        processed = 0
        for wiki_file in wiki_files:
            try:
                with open(wiki_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            article = json.loads(line.strip())
                            title = article.get('title', '').strip()
                            text = article.get('text', '').strip()

                            if not title or not text:
                                continue

                            # Normalize title (lowercase for matching)
                            title_lower = title.lower()

                            # Extract summary (first paragraph or first 2 sentences)
                            summary = self._extract_summary(text)

                            if summary:
                                index[title_lower] = {
                                    'title': title,  # Original case
                                    'summary': summary
                                }
                                processed += 1

                                if processed % 10000 == 0:
                                    logger.info(f"Processed {processed} articles...")

                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.debug(f"Error processing line: {e}")
                            continue

            except Exception as e:
                logger.error(f"Error processing file {wiki_file}: {e}")
                continue

        logger.info(f"Index built with {len(index)} articles")

        # Save index
        try:
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_file, 'wb') as f:
                pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Index saved to {self.index_file}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

        return index

    def _extract_summary(self, text: str, max_sentences: int = 3) -> str:
        """
        Extract summary from article text.

        Strategy:
        1. Take first paragraph (before first \n\n)
        2. If too long, take first N sentences
        3. Clean up formatting
        """
        # Remove section headers and formatting
        text = re.sub(r'\n==.*?==\n', '\n', text)
        text = re.sub(r'\n\n+', '\n\n', text)

        # Get first paragraph
        paragraphs = text.split('\n\n')
        first_para = paragraphs[0] if paragraphs else text

        # Split into sentences (simple heuristic)
        sentences = re.split(r'(?<=[.!?])\s+', first_para)

        # Take first N sentences
        summary_sentences = sentences[:max_sentences]
        summary = ' '.join(summary_sentences).strip()

        # Clean up
        summary = re.sub(r'\s+', ' ', summary)  # Normalize whitespace
        summary = re.sub(r'\[\d+\]', '', summary)  # Remove citation markers

        return summary

    def get_summary(self, term: str, sentences: int = 2) -> Optional[str]:
        """
        Get summary for a term.

        Args:
            term: Term to look up
            sentences: Number of sentences to return

        Returns:
            Summary text or None if not found
        """
        if not term:
            return None

        term_lower = term.lower().strip()

        # Check cache first
        cache_key = (term_lower, sentences)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        self.cache_misses += 1

        # Exact match
        if term_lower in self.index:
            summary = self._truncate_summary(
                self.index[term_lower]['summary'],
                sentences
            )
            self._add_to_cache(cache_key, summary)
            return summary

        # Fuzzy match (try variations)
        variations = self._generate_variations(term_lower)
        for variation in variations:
            if variation in self.index:
                summary = self._truncate_summary(
                    self.index[variation]['summary'],
                    sentences
                )
                self._add_to_cache(cache_key, summary)
                return summary

        # Partial match (contains term)
        for key in self.index:
            if term_lower in key or key in term_lower:
                summary = self._truncate_summary(
                    self.index[key]['summary'],
                    sentences
                )
                self._add_to_cache(cache_key, summary)
                return summary

        return None

    def _generate_variations(self, term: str) -> List[str]:
        """Generate term variations for fuzzy matching."""
        variations = [term]

        # Plural/singular
        if term.endswith('s'):
            variations.append(term[:-1])
        else:
            variations.append(term + 's')

        # With/without articles
        for article in ['the ', 'a ', 'an ']:
            if term.startswith(article):
                variations.append(term[len(article):])
            else:
                variations.append(article + term)

        # Capitalization variations
        variations.append(term.capitalize())
        variations.append(term.title())

        return variations

    def _truncate_summary(self, summary: str, sentences: int) -> str:
        """Truncate summary to N sentences."""
        if not summary:
            return ""

        # Split into sentences
        sentence_list = re.split(r'(?<=[.!?])\s+', summary)

        # Take first N
        truncated = ' '.join(sentence_list[:sentences])

        # Ensure ends with punctuation
        if truncated and not truncated[-1] in '.!?':
            truncated += '.'

        return truncated

    def _add_to_cache(self, key: Tuple, value: str):
        """Add to cache with LRU eviction."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO for now)
            self.cache.pop(next(iter(self.cache)))

        self.cache[key] = value

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str]]:
        """
        Search for articles matching query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (title, summary) tuples
        """
        query_lower = query.lower().strip()
        results = []

        # Exact matches first
        if query_lower in self.index:
            entry = self.index[query_lower]
            results.append((entry['title'], entry['summary']))

        # Partial matches
        for key, entry in self.index.items():
            if len(results) >= top_k:
                break

            if query_lower in key and (entry['title'], entry['summary']) not in results:
                results.append((entry['title'], entry['summary']))

        return results[:top_k]

    def get_stats(self) -> Dict:
        """Get statistics about the index and cache."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            'total_articles': len(self.index),
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': hit_rate,
            'index_file': str(self.index_file),
            'index_file_size_mb': self.index_file.stat().st_size / (1024 * 1024) if self.index_file.exists() else 0
        }

    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


# Convenience function for quick access
_global_wiki = None

def get_wikipedia(wiki_dir: str = "data/wikipedia/extracted") -> LocalWikipedia:
    """Get global Wikipedia instance (singleton)."""
    global _global_wiki
    if _global_wiki is None:
        _global_wiki = LocalWikipedia(wiki_dir)
    return _global_wiki
