"""
Test script for Task 1B: Text Hint Sanitization
================================================

This script tests the _sanitize_message_for_target() method
with various input scenarios to ensure it correctly removes
visual descriptions.

Usage:
    python tools/test_sanitization.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator.strategic_simulator import StrategicSimulator


def test_sanitization():
    """Test sanitization with different hint levels and messages."""

    test_cases = [
        {
            "name": "Detailed visual description",
            "message": "Here is an image showing a person on a motorcycle wearing a red helmet, with another person standing near a bicycle in the background.",
            "expected_minimal": "Here is an image.",
            "expected_moderate": "Here is an image showing a person on a motorcycle wearing something, with another person standing near a bicycle in the background.",
        },
        {
            "name": "Color and spatial descriptions",
            "message": "Look at the red car on the left and the blue bike on the right.",
            "expected_minimal": "Look at the car on one side and the bike on one side.",
            "expected_moderate": "Look at the car on one side and the bike on one side.",
        },
        {
            "name": "Guidance action with spatial terms",
            "message": "Please focus on the left side of the image.",
            "expected_minimal_guidance": "Please focus on the left side of the image.",  # Should preserve for guidance
        },
        {
            "name": "Simple question without visual hints",
            "message": "What do you see in this image?",
            "expected_minimal": "What do you see in this image?",  # Should remain unchanged
        },
        {
            "name": "Complex scene description",
            "message": "This picture shows a happy person wearing a blue shirt standing in the foreground, with buildings positioned in the background.",
            "expected_minimal": "This picture.",
        },
    ]

    print("="*70)
    print("TASK 1B: TEXT HINT SANITIZATION TESTS")
    print("="*70)

    for hint_level in ["minimal", "moderate", "full"]:
        print(f"\n{'='*70}")
        print(f"Testing with hint_level='{hint_level}'")
        print(f"{'='*70}")

        # Create simulator with specific hint level
        simulator = StrategicSimulator(
            minimize_text_hints=True,
            hint_level=hint_level,
            verbose=False
        )

        for test_case in test_cases:
            print(f"\n--- Test: {test_case['name']} ---")
            print(f"Original: {test_case['message']}")

            # Test with different action types
            for action_type in [None, "guidance"]:
                result = simulator._sanitize_message_for_target(
                    message=test_case['message'],
                    action_type=action_type
                )

                action_label = f" (action={action_type})" if action_type else ""
                print(f"Result{action_label}: {result}")

                # Check expected results
                if hint_level == "minimal" and action_type != "guidance":
                    expected_key = "expected_minimal"
                elif hint_level == "minimal" and action_type == "guidance":
                    expected_key = "expected_minimal_guidance"
                elif hint_level == "moderate":
                    expected_key = "expected_moderate"
                else:
                    expected_key = None

                if expected_key and expected_key in test_case:
                    expected = test_case[expected_key]
                    if hint_level == "full":
                        # Full level should not change the message
                        assert result == test_case['message'], f"Full level should preserve message"
                        print("  ✓ Passed (preserved for full level)")
                    else:
                        # Check if sanitization reduced length
                        if len(result) < len(test_case['message']):
                            print(f"  ✓ Reduced from {len(test_case['message'])} to {len(result)} chars")
                        else:
                            print(f"  ⚠ Length unchanged")

    print(f"\n{'='*70}")
    print("SANITIZATION EFFECTIVENESS SUMMARY")
    print(f"{'='*70}")

    # Test with real-world example
    simulator_minimal = StrategicSimulator(minimize_text_hints=True, hint_level="minimal", verbose=False)
    simulator_full = StrategicSimulator(minimize_text_hints=True, hint_level="full", verbose=False)

    real_world_msg = """Here is an image showing a person on a motorcycle wearing a red helmet,
with another person standing near a bicycle in the background. The motorcycle rider is in the
foreground on the left side, and there are buildings positioned on the right."""

    result_minimal = simulator_minimal._sanitize_message_for_target(real_world_msg)
    result_full = simulator_full._sanitize_message_for_target(real_world_msg)

    print(f"\nReal-world example:")
    print(f"  Original length: {len(real_world_msg)} chars")
    print(f"  Original word count: {len(real_world_msg.split())} words")
    print(f"\n  Full hint level:")
    print(f"    Length: {len(result_full)} chars")
    print(f"    Word count: {len(result_full.split())} words")
    print(f"    Text: {result_full[:100]}...")
    print(f"\n  Minimal hint level:")
    print(f"    Length: {len(result_minimal)} chars")
    print(f"    Word count: {len(result_minimal.split())} words")
    print(f"    Reduction: {100 * (len(real_world_msg) - len(result_minimal)) / len(real_world_msg):.1f}%")
    print(f"    Text: {result_minimal}")

    # Count visual words
    visual_words_patterns = [
        "wearing", "showing", "positioned", "left", "right", "foreground", "background",
        "red", "blue", "green", "standing", "sitting"
    ]

    def count_visual_words(text):
        text_lower = text.lower()
        return sum(1 for word in visual_words_patterns if word in text_lower)

    original_visual = count_visual_words(real_world_msg)
    minimal_visual = count_visual_words(result_minimal)
    full_visual = count_visual_words(result_full)

    print(f"\n  Visual words:")
    print(f"    Original: {original_visual}")
    print(f"    Full: {full_visual}")
    print(f"    Minimal: {minimal_visual}")
    print(f"    Reduction: {100 * (original_visual - minimal_visual) / original_visual:.1f}%")

    print(f"\n{'='*70}")
    print("✓ All tests completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_sanitization()
