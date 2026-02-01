"""
Tool to check text hints in user simulator messages (Task 1B validation)
=========================================================================

This script analyzes simulator log files to count visual descriptive words
that could leak information to the target model without requiring vision.

Usage:
    python tools/check_text_hints.py test_output/run_log_20240101_120000.json
    python tools/check_text_hints.py test_output/run_log_*.json
"""

import json
import re
from pathlib import Path
import sys
from typing import Dict, List


# Visual description patterns to detect
VISUAL_PATTERNS = [
    # Descriptive verbs
    (r'\b(wearing|showing|positioned|colored|located)\b', 'descriptive_verbs'),
    # Spatial terms
    (r'\b(left|right|center|top|bottom|foreground|background)\b', 'spatial_terms'),
    # Colors
    (r'\b(red|blue|green|yellow|black|white|brown|gray|grey|orange|purple|pink)\b', 'colors'),
    # Size/quantity descriptors
    (r'\b(big|small|large|tiny|huge|enormous|many|few|several)\b', 'size_quantity'),
    # State descriptors
    (r'\b(open|closed|standing|sitting|lying|running|walking)\b', 'state_descriptors'),
    # Emotion descriptors
    (r'\b(happy|sad|angry|smiling|frowning|excited)\b', 'emotion_descriptors'),
]


def count_visual_words(message: str) -> Dict[str, int]:
    """
    Count words that describe visual content.

    Returns:
        Dict mapping category to count
    """
    counts = {}
    total = 0

    for pattern, category in VISUAL_PATTERNS:
        matches = re.findall(pattern, message, re.IGNORECASE)
        count = len(matches)
        counts[category] = count
        total += count

    counts['total'] = total
    return counts


def analyze_log_file(log_path: Path) -> Dict:
    """
    Analyze a single log file for text hints.

    Returns:
        Statistics about text hints in the log
    """
    with open(log_path, encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing {log_path}: {e}")
            return {}

    # Handle different log formats
    if isinstance(data, dict):
        # New format with run_log key
        events = data.get('run_log', [])
    elif isinstance(data, list):
        # Old format - direct list of events
        events = data
    else:
        print(f"Unknown log format in {log_path}")
        return {}

    # Find user messages (turns)
    user_turns = []
    for event in events:
        if event.get('event') == 'turn':
            user_turns.append(event.get('data', {}).get('message', ''))
        elif event.get('event') == 'core_model_decision':
            # Alternative format
            user_turns.append(event.get('user_query', ''))

    if not user_turns:
        print(f"No user turns found in {log_path}")
        return {}

    # Analyze each turn
    total_visual_words = 0
    total_words = 0
    category_totals = {}
    turn_details = []

    for i, message in enumerate(user_turns):
        if not message:
            continue

        visual_counts = count_visual_words(message)
        word_count = len(message.split())

        total_visual_words += visual_counts['total']
        total_words += word_count

        for category, count in visual_counts.items():
            if category != 'total':
                category_totals[category] = category_totals.get(category, 0) + count

        turn_details.append({
            'turn': i + 1,
            'visual_words': visual_counts['total'],
            'total_words': word_count,
            'visual_ratio': visual_counts['total'] / word_count if word_count > 0 else 0,
            'message_preview': message[:80] + '...' if len(message) > 80 else message
        })

    avg_visual_words = total_visual_words / len(user_turns) if user_turns else 0
    avg_total_words = total_words / len(user_turns) if user_turns else 0
    visual_word_ratio = total_visual_words / total_words if total_words > 0 else 0

    return {
        'log_file': log_path.name,
        'total_turns': len(user_turns),
        'total_visual_words': total_visual_words,
        'total_words': total_words,
        'avg_visual_words_per_turn': avg_visual_words,
        'avg_words_per_turn': avg_total_words,
        'visual_word_ratio': visual_word_ratio,
        'category_breakdown': category_totals,
        'turn_details': turn_details
    }


def print_report(stats: Dict):
    """Print a formatted report of text hint statistics."""
    print(f"\n{'='*60}")
    print(f"Log: {stats['log_file']}")
    print(f"{'='*60}")
    print(f"Total user turns: {stats['total_turns']}")
    print(f"Total visual words: {stats['total_visual_words']}")
    print(f"Total words: {stats['total_words']}")
    print(f"Avg visual words/turn: {stats['avg_visual_words_per_turn']:.2f}")
    print(f"Avg words/turn: {stats['avg_words_per_turn']:.2f}")
    print(f"Visual word ratio: {stats['visual_word_ratio']:.2%}")

    print(f"\nCategory Breakdown:")
    for category, count in sorted(stats['category_breakdown'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {category:20s}: {count:3d}")

    # Pass/Fail criteria
    target_visual_words = 2  # Target: <2 visual words per turn
    status = "✓ PASS" if stats['avg_visual_words_per_turn'] < target_visual_words else "✗ FAIL"

    print(f"\n{'='*60}")
    print(f"Target: <{target_visual_words} visual words/turn")
    print(f"Status: {status}")
    print(f"{'='*60}")

    # Show worst offenders (turns with most visual words)
    worst_turns = sorted(stats['turn_details'], key=lambda x: x['visual_words'], reverse=True)[:5]
    if worst_turns[0]['visual_words'] > 0:
        print(f"\nTop 5 turns with most visual words:")
        for turn in worst_turns:
            print(f"  Turn {turn['turn']}: {turn['visual_words']} visual words ({turn['visual_ratio']:.1%})")
            print(f"    \"{turn['message_preview']}\"")


def compare_logs(log_paths: List[Path]):
    """Compare multiple log files (e.g., minimal vs full hints)."""
    all_stats = []

    for log_path in log_paths:
        stats = analyze_log_file(log_path)
        if stats:
            all_stats.append(stats)

    if len(all_stats) < 2:
        print("Need at least 2 log files for comparison")
        return

    print(f"\n{'='*60}")
    print("COMPARISON REPORT")
    print(f"{'='*60}")
    print(f"{'Log File':<30s} {'Avg Visual/Turn':>15s} {'Ratio':>10s}")
    print(f"{'-'*60}")

    for stats in all_stats:
        print(f"{stats['log_file']:<30s} {stats['avg_visual_words_per_turn']:>15.2f} {stats['visual_word_ratio']:>10.1%}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_text_hints.py <log_file.json> [<log_file2.json> ...]")
        print("\nExample:")
        print("  python tools/check_text_hints.py test_output/run_log_20240101.json")
        print("  python tools/check_text_hints.py test_output/minimal/*.json test_output/full/*.json")
        sys.exit(1)

    log_paths = []
    for arg in sys.argv[1:]:
        # Support glob patterns
        path = Path(arg)
        if '*' in arg:
            # Glob pattern
            parent = Path(arg).parent
            pattern = Path(arg).name
            log_paths.extend(parent.glob(pattern))
        elif path.exists():
            log_paths.append(path)
        else:
            print(f"File not found: {arg}")

    if not log_paths:
        print("No valid log files found")
        sys.exit(1)

    # Analyze each log
    for log_path in log_paths:
        stats = analyze_log_file(log_path)
        if stats:
            print_report(stats)

    # If multiple logs, show comparison
    if len(log_paths) > 1:
        compare_logs(log_paths)


if __name__ == "__main__":
    main()
