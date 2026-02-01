"""
Quick test script to verify action diversity before full experiment run.
"""

from src.simulator.action_selector import ActionSelector
from collections import Counter

def test_action_selector():
    """Test if action selector generates diverse actions"""
    print("=" * 60)
    print("Testing Action Selector Diversity")
    print("=" * 60)

    selector = ActionSelector(strategy="rule_based")

    # Simulate 10 turns
    actions_by_turn = []
    for turn in range(10):
        action = selector.select(
            turn=turn,
            history=[{"action": actions_by_turn[-1]} if actions_by_turn else {}],
            entities={},
            task_info={}
        )
        actions_by_turn.append(action)
        print(f"Turn {turn}: {action}")

    # Analyze diversity
    action_counts = Counter(actions_by_turn)
    unique_count = len(action_counts)

    print("\n" + "=" * 60)
    print("Action Distribution (10 turns):")
    print("=" * 60)
    for action, count in action_counts.most_common():
        pct = count / len(actions_by_turn) * 100
        print(f"  {action:25s}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print(f"Unique actions: {unique_count}")

    if unique_count >= 6:
        print(f"✅ GOOD: {unique_count} unique actions in 10 turns")
    else:
        print(f"⚠️  WARN: Only {unique_count} unique actions in 10 turns")

    max_pct = max(c / len(actions_by_turn) for c in action_counts.values()) * 100
    if max_pct <= 50:
        print(f"✅ GOOD: Max action percentage {max_pct:.1f}% (should be < 50% for 10 turns)")
    else:
        print(f"⚠️  WARN: Max action percentage {max_pct:.1f}%")

    print("=" * 60)

def test_action_selector_multiple_runs():
    """Test action selector over multiple runs to see overall distribution"""
    print("\n" + "=" * 60)
    print("Testing Action Selector - 100 turns across 10 runs")
    print("=" * 60)

    selector = ActionSelector(strategy="rule_based")

    all_actions = []
    for run in range(10):
        for turn in range(10):
            action = selector.select(
                turn=turn,
                history=[],
                entities={},
                task_info={}
            )
            all_actions.append(action)

    action_counts = Counter(all_actions)
    unique_count = len(action_counts)

    print(f"\nTotal actions: {len(all_actions)}")
    print(f"Unique actions: {unique_count}")

    print("\n--- Action Distribution (100 turns) ---")
    for action, count in action_counts.most_common():
        pct = count / len(all_actions) * 100
        bar = "█" * int(pct / 2)
        print(f"  {action:25s}: {count:3d} ({pct:5.1f}%) {bar}")

    print("\n" + "=" * 60)
    if unique_count >= 8:
        print(f"✅ PASS: {unique_count} unique actions (>= 8)")
    else:
        print(f"❌ FAIL: Only {unique_count} unique actions (need >= 8)")

    max_pct = max(c / len(all_actions) for c in action_counts.values()) * 100
    max_action = action_counts.most_common(1)[0][0]
    if max_pct <= 40:
        print(f"✅ PASS: Max action '{max_action}' at {max_pct:.1f}% (<= 40%)")
    else:
        print(f"⚠️  WARN: Max action '{max_action}' at {max_pct:.1f}% (target <= 40%)")

    print("=" * 60)

if __name__ == "__main__":
    test_action_selector()
    test_action_selector_multiple_runs()
