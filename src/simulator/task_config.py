"""
Task Configuration for M3Bench User Simulator
==============================================

Defines task types, prompts, and evaluation criteria for each task.
"""

# ========== Task Type Definitions ==========

TASK_CONFIGS = {
    # ---------- Attribute Comparison (AC) ----------
    "attribute_comparison": {
        "name": "Attribute Comparison",
        "name_zh": "属性比较",
        "description": """
This task tests the model's ability to compare object attributes across multiple images.
The model needs to:
1. Identify the same type of object in different images
2. Compare specific attributes (count, size, position, color)
3. Give the correct answer about which image has the target attribute

Example: "Which image contains the most people?" - The model should count people in each image and identify the correct one.
""",
        "goal": "Compare object attributes across images and identify the correct answer",
        "evaluation_criteria": [
            "Correctly identifies objects in all images",
            "Accurately compares the specified attribute",
            "Provides the correct answer with reasoning",
            "Shows evidence-based reasoning (not guessing)"
        ],
        "multi_turn_strategy": {
            "phase_1": "Present images one by one with brief descriptions",
            "phase_2": "Ask the comparison question after all images are shown",
            "phase_3": "Challenge or verify the answer if uncertain",
            "noise_injection": "Introduce irrelevant details or misleading hints"
        },
        "action_weights": {
            "guidance": 0.3,      # Guide to look at specific images
            "follow_up": 0.25,    # Ask for more details
            "mislead": 0.15,      # Test robustness to wrong hints
            "fine_grained": 0.2,  # Ask for precise counts/measurements
            "distraction": 0.1   # Test attention filtering
        }
    },

    # ---------- Visual Noise Filtering (VNF) ----------
    "visual_noise_filtering": {
        "name": "Visual Noise Filtering",
        "name_zh": "视觉噪声过滤",
        "description": """
This task tests the model's ability to filter relevant information from noise.
The user will:
1. Inject irrelevant information during conversation (noise)
2. Mix relevant and irrelevant questions
3. Finally ask a question that requires filtering noise to answer correctly

The model needs to:
1. Track relevant information across turns
2. Ignore irrelevant/misleading information
3. Answer based on actual visual evidence, not injected noise
""",
        "goal": "Filter noise and answer based on visual evidence",
        "evaluation_criteria": [
            "Maintains focus on relevant information",
            "Does not get confused by injected noise",
            "Correctly recalls relevant details after noise",
            "Answer matches visual evidence, not noise"
        ],
        "multi_turn_strategy": {
            "phase_1": "Establish baseline understanding of the image",
            "phase_2": "Inject noise (irrelevant questions, wrong info)",
            "phase_3": "Ask the target question requiring noise filtering",
            "noise_types": ["irrelevant_details", "wrong_attributes", "confusing_references"]
        },
        "action_weights": {
            "distraction": 0.3,    # Inject irrelevant info
            "mislead": 0.25,       # Inject wrong info
            "redundancy": 0.15,    # Repeat to confuse
            "follow_up": 0.2,      # Mix with relevant questions
            "fine_grained": 0.1    # Final precise question
        }
    },

    # ---------- Attribute Bridge Reasoning (ABR) ----------
    "attribute_bridge_reasoning": {
        "name": "Attribute Bridge Reasoning",
        "name_zh": "属性桥接推理",
        "description": """
This task tests multi-hop reasoning through object relationships.
The model needs to:
1. Identify objects and their attributes
2. Follow relationship chains (A relates to B, B relates to C)
3. Answer questions that require bridging multiple relations

Example: "Find the object left of the red car. What color is the object above it?"
- Requires: finding red car -> finding object left of it -> finding object above that -> getting its color
""",
        "goal": "Perform multi-hop reasoning through object relationships",
        "evaluation_criteria": [
            "Correctly identifies initial object",
            "Follows relationship chain correctly",
            "Each reasoning step is logically valid",
            "Final answer is correct and justified"
        ],
        "multi_turn_strategy": {
            "phase_1": "Identify key objects in the image",
            "phase_2": "Establish relationships between objects",
            "phase_3": "Ask multi-hop reasoning question",
            "phase_4": "Verify reasoning steps if answer seems wrong"
        },
        "action_weights": {
            "follow_up": 0.35,     # Guide through reasoning steps
            "guidance": 0.25,      # Point to relevant regions
            "negation": 0.15,      # Correct wrong reasoning
            "fine_grained": 0.15,  # Ask for precise details
            "logic_skip": 0.1      # Test if model refuses to skip steps
        }
    },

    # ---------- Relation Comparison (RC) ----------
    "relation_comparison": {
        "name": "Relation Comparison",
        "name_zh": "关系比较",
        "description": """
This task compares relationships between objects across images.
The model needs to:
1. Identify relationships in multiple images
2. Compare the nature or type of relationships
3. Find similarities or differences

Example: "In which image are the people more closely interacting?"
""",
        "goal": "Compare object relationships across images",
        "evaluation_criteria": [
            "Correctly identifies relationships in each image",
            "Valid comparison logic",
            "Answer supported by evidence"
        ],
        "multi_turn_strategy": {
            "phase_1": "Present images and establish relationships",
            "phase_2": "Ask for relationship comparison",
            "phase_3": "Verify with specific examples"
        },
        "action_weights": {
            "follow_up": 0.3,
            "guidance": 0.25,
            "fine_grained": 0.2,
            "distraction": 0.15,
            "mislead": 0.1
        }
    }
}


# ========== Core Model System Prompt ==========

CORE_MODEL_SYSTEM_PROMPT = """You are a user simulator for testing Vision-Language Models (VLMs).

Your role is to simulate a natural human user who:
1. Has a specific task goal (defined below)
2. Presents information gradually across multiple turns
3. Injects noise/distractions naturally
4. Evaluates the model's responses
5. Decides when to move to the next task

## Current Task Information
{task_info}

## Task Goal
{task_goal}

## Available Actions
You must choose ONE of these actions each turn:

1. **guidance**: Point the model to look at specific regions or aspects
   - Example: "Look at the left side of the image" or "Notice the red object"

2. **follow_up**: Ask for more details or clarification
   - Example: "Can you describe that in more detail?" or "What about its color?"

3. **mislead**: Inject wrong information to test robustness
   - Example: "I think there are 5 people, right?" (when there are only 3)

4. **distraction**: Inject irrelevant information or questions
   - Example: "By the way, what's the weather like?" or "That reminds me of something else..."

5. **fine_grained**: Ask for precise details
   - Example: "Exactly how many?" or "Can you give the precise location?"

6. **negation**: Correct wrong reasoning
   - Example: "That's not right, look again at..."

7. **redundancy**: Repeat information (to test memory)
   - Example: "You mentioned X earlier, can you confirm that again?"

8. **next_task**: Move to the next task (only when current task is complete or failed)

## Memory Store
{memory_content}

## Output Format
You must respond in the following JSON format:
```json
{{
    "action": "action_name",
    "message_to_model": "Your natural language message to the VLM",
    "reasoning": "Why you chose this action",
    "evaluation_of_last_response": "Your assessment of the model's last response (if any)",
    "key_info_to_remember": ["list", "of", "important", "info", "to", "save"],
    "task_progress": "incomplete/partial/complete/failed",
    "confidence": 0.0-1.0
}}
```

## Guidelines
1. Be natural - simulate a real user, not a robot
2. Follow the multi-turn strategy for this task type
3. Only use 'next_task' when the task is clearly complete or failed
4. Extract and remember key information from model responses
5. Be fair in evaluation - give credit where due
"""


# ========== Response Evaluation Prompts ==========

EVALUATION_PROMPT = """
Evaluate the VLM's response based on these criteria:

Task: {task_type}
Question Asked: {question}
Model Response: {response}
Expected Answer: {expected_answer}

Evaluation Criteria:
{criteria}

Rate on scale 1-5:
1 = Completely wrong
2 = Mostly wrong, some correct elements
3 = Partially correct
4 = Mostly correct, minor issues
5 = Completely correct

Output format:
```json
{{
    "score": 1-5,
    "correct_elements": ["list of correct aspects"],
    "wrong_elements": ["list of incorrect aspects"],
    "reasoning": "explanation of evaluation"
}}
```
"""


# ========== Action Templates by Task Type ==========

TASK_ACTION_TEMPLATES = {
    "attribute_comparison": {
        "guidance": [
            "Take a look at Image {image_idx}.",
            "Now let's focus on Image {image_idx}. What do you see?",
            "Here's another image to consider. [Shows Image {image_idx}]"
        ],
        "follow_up": [
            "How many {object_type} can you count in this image?",
            "Can you describe the {attribute} of the {object_type}?",
            "What's different about this image compared to the previous one?"
        ],
        "mislead": [
            "I think Image {wrong_idx} has more {object_type}, don't you agree?",
            "The {attribute} in Image {wrong_idx} seems larger to me.",
            "Are you sure? It looks like Image {wrong_idx} to me."
        ],
        "final_question": [
            "Now that you've seen all images, {question}",
            "Based on what you observed, {question}",
            "Comparing all the images, {question}"
        ]
    },

    "visual_noise_filtering": {
        "noise_injection": [
            "By the way, did you notice the weather in the background?",
            "That reminds me, what time of day do you think this is?",
            "Speaking of which, is there anything unusual about the lighting?",
            "Oh, I forgot to mention - what about the shadows?"
        ],
        "misleading_info": [
            "I remember there being a {wrong_object} in the image.",
            "Earlier you said there were {wrong_count} {object_type}, right?",
            "The {object_type} was {wrong_attribute}, wasn't it?"
        ],
        "final_question": [
            "Ignoring everything else, {question}",
            "Based only on what you actually see, {question}",
            "Let me ask the important question: {question}"
        ]
    },

    "attribute_bridge_reasoning": {
        "step_by_step": [
            "First, can you find the {object_1}?",
            "Good. Now what's {relation} the {object_1}?",
            "And what about {next_relation} that object?",
            "Finally, what is the {attribute} of that object?"
        ],
        "verification": [
            "Let me verify - you said the chain is: {chain}. Is that correct?",
            "Can you walk me through your reasoning again?",
            "I want to make sure - {intermediate_step}, right?"
        ]
    }
}