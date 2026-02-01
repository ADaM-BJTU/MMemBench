"""
Context Padding with Weak Model Collaboration
==============================================

Implements "pseudo multi-turn" conversations using a weak/cheap model
to generate filler content that extends context length without burning
expensive tokens on the main model.

Purpose:
- Test long-context capabilities
- Create realistic conversation length
- Save costs by using cheap model for verbose content
"""

import random
import requests
import json
import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FillerConfig:
    """Configuration for filler generation"""
    min_filler_turns: int = 3
    max_filler_turns: int = 8
    filler_position: str = "distributed"  # "early", "middle", "late", "distributed"
    min_filler_length: int = 100  # Minimum characters per filler response
    max_filler_length: int = 500  # Maximum characters per filler response
    topics: List[str] = None

    def __post_init__(self):
        if self.topics is None:
            self.topics = [
                "background_details",
                "scene_description",
                "tangential_observation",
                "user_commentary",
                "verbose_confirmation",
                "extended_question",
                "contextual_rambling"
            ]


class ContextPadder:
    """
    Generates filler content to pad conversation context.
    Uses a weak model for cost efficiency with longer, more realistic content.
    """

    # Pre-defined filler templates for fallback
    FILLER_USER_TEMPLATES = {
        "background_details": [
            "顺便说一下，这张图片的背景看起来很有意思。你能详细描述一下背景中的所有元素吗？包括颜色、纹理、光线等各个方面。",
            "我注意到背景里有一些细节，比如光影、色调、空间布局等。你能详细分析一下吗？我想了解更多。",
            "The background of this image is quite interesting. Can you describe in detail all the elements you see? Include colors, textures, lighting, and spatial arrangement.",
            "Let's take a deep dive into the background elements. What can you tell me about the environment, the lighting conditions, and any patterns you notice?",
        ],
        "scene_description": [
            "这个场景给你什么整体感觉？从构图、色彩搭配、主体位置、光线方向等多个角度来分析一下。我想听听你的详细看法。",
            "请从艺术欣赏的角度来描述一下这张图片。谈谈它的构图美学、色彩运用、以及你认为摄影师想要表达的情感。",
            "What's the overall mood and atmosphere of this image? Analyze it from multiple perspectives: composition, color palette, subject positioning, and lighting direction.",
            "Can you give me a comprehensive analysis of this scene? Include your thoughts on the artistic elements, the emotional tone, and what story this image might be telling.",
        ],
        "tangential_observation": [
            "说说图中任何你觉得有趣或不寻常的细节。哪怕是很小的东西，也请详细描述一下，我对所有细节都很感兴趣。",
            "有没有什么你一开始没注意到、但仔细看才发现的元素？请分享一下你的观察过程和发现。",
            "Is there anything unusual, unexpected, or particularly noteworthy in this image? Please describe in detail, even small things. I'm interested in all the nuances.",
            "What catches your eye when you look at this image? Walk me through your observation process - what did you notice first, and what did you discover upon closer inspection?",
        ],
        "user_commentary": [
            "这张图让我想起了一些事情...你觉得这种类型的场景在现实生活中常见吗？能不能结合你的理解谈谈？",
            "我觉得这张图片挺有意思的，尤其是那些细微的地方。你有什么独特的见解想分享吗？请尽量详细说说。",
            "This image reminds me of something. Do you think this type of scene is common in real life? Can you elaborate on your understanding and share your perspective in detail?",
            "I find this image quite fascinating, especially the subtle details. Do you have any unique insights to share? Please be as detailed as possible.",
        ],
        "verbose_confirmation": [
            "让我再详细确认一下之前讨论的内容。你之前提到了一些观察，能不能把所有要点重新完整地叙述一遍？我想确保我没有遗漏任何信息。",
            "我想做一个全面的回顾。请把目前为止我们讨论过的所有关键信息都总结一下，越详细越好。",
            "Let me thoroughly confirm what we discussed earlier. Can you restate all the key points in full detail? I want to make sure I haven't missed anything important.",
            "I'd like to do a comprehensive review. Please summarize all the key information we've discussed so far, as detailed as possible.",
        ],
        "extended_question": [
            "关于图中的主要元素，能从以下几个方面详细说说吗：1)它的准确位置和大小，2)颜色和材质特征，3)与周围环境的关系，4)可能的功能或意义。",
            "请对图像进行一个系统性的描述，包括：前景、中景、背景分别有什么；各个元素之间的空间关系；整体的视觉层次感。",
            "Regarding the main elements in this image, can you elaborate on: 1) their exact position and size, 2) color and texture characteristics, 3) relationship with surroundings, 4) possible function or significance.",
            "Please provide a systematic description of the image, including: what's in the foreground, middle ground, and background; spatial relationships between elements; overall visual hierarchy.",
        ],
        "contextual_rambling": [
            "你知道吗，我最近一直在研究图像分析。关于这张图，我想听听你从不同角度的分析——技术角度、艺术角度、还有日常生活角度。请畅所欲言。",
            "我对这类图像很感兴趣，因为它们往往包含很多信息。你能不能像讲故事一样，把图中发生的事情详细描述出来？",
            "You know, I've been studying image analysis lately. For this image, I'd like to hear your analysis from different perspectives - technical, artistic, and everyday life. Please share your thoughts freely.",
            "I'm fascinated by images like this because they often contain rich information. Could you describe what's happening in this image as if you were telling a story? Include all the details you can observe.",
        ],
    }

    # LLM prompts for generating verbose filler
    WEAK_MODEL_SYSTEM_PROMPT = """你是一个对话助手，正在和用户讨论一张图片。请用详细、自然、略带冗长的方式回答问题。

要求：
1. 回答要详细，至少150-300字
2. 可以适当重复或用不同方式表达同一观点
3. 使用自然的口语化表达
4. 可以加入一些"我认为"、"如果我没看错"等不确定性表达
5. 适当进行延伸讨论，但不要偏离主题太远

请确保回复足够详细，因为这是一个测试长对话能力的场景。"""

    WEAK_MODEL_USER_TEMPLATE = """用户正在和一个视觉模型讨论图片。请生成一段详细的回复来回答以下问题。

话题类型：{topic}
用户问题：{question}
之前的讨论要点：{context}

请生成一段{min_length}-{max_length}字的详细回复，风格自然，略带冗长。"""

    # 新增：多图场景的filler模板
    MULTI_IMAGE_FILLER_USER_TEMPLATES = {
        "cross_image_comparison": [
            "让我们比较一下这几张图片。在{image_ref_1}和{image_ref_2}之间，{object}有什么不同之处？请详细说明。",
            "仔细看看{image_ref_1}和{image_ref_2}，你能找出{object}在两张图中的所有差异吗？",
            "Compare the {object} in {image_ref_1} and {image_ref_2}. What differences can you identify?",
            "Looking at both {image_ref_1} and {image_ref_2}, describe how the {object} differs between them.",
        ],
        "image_detail_expansion": [
            "关于{image_ref}中的{object}，请从颜色、大小、位置、状态等多个角度进行详细描述。",
            "请深入分析{image_ref}中{object}的所有可见特征，包括但不限于外观、环境、和周围物体的关系。",
            "For the {object} in {image_ref}, provide a comprehensive description covering color, size, position, and condition.",
            "Analyze all visible features of the {object} in {image_ref}, including appearance, environment, and relationships.",
        ],
        "object_tracking": [
            "我们已经讨论了多张图中的{object}。能不能帮我整理一下，每张图里的{object}分别有什么特点？",
            "回顾一下我们看过的几张图，{object}在不同图片中分别是什么样的？请逐一说明。",
            "Let's review the {object} across the images we've seen. What are the distinct characteristics in each?",
            "Summarize the {object} features from each image we discussed so far.",
        ],
        "tangential_multi_image": [
            "说到{image_ref}中的{object}，我想起来{image_ref_2}里好像也有类似的东西？它们有关联吗？",
            "在{image_ref}讨论这个{object}的时候，我注意到它和{image_ref_2}的{related_object}可能有关系。你怎么看？",
            "While looking at the {object} in {image_ref}, I noticed something similar in {image_ref_2}. Are they related?",
            "The {object} in {image_ref} reminds me of the {related_object} in {image_ref_2}. What's your take?",
        ],
        "memory_refresh": [
            "在我们继续之前，能不能再确认一下：{image_ref}里的{object}具体是什么样的？我想确保我记得正确。",
            "我有点混淆了。{image_ref}里那个{object}，它的{attribute}是什么来着？",
            "Before we continue, can you remind me: what exactly does the {object} in {image_ref} look like?",
            "I'm getting a bit confused. The {object} in {image_ref}, what was its {attribute} again?",
        ],
    }

    # 多图场景的回复模板（用于模板回复生成）
    MULTI_IMAGE_RESPONSE_TEMPLATES = {
        "cross_image_comparison": [
            "让我仔细比较一下这两张图中的{object}。在{image_ref_1}中，{object}呈现出{attr_1}的特点，"
            "而在{image_ref_2}中，{object}则显示为{attr_2}。这种差异可能是由于{reason}造成的。"
            "从整体来看，两者在{aspect}方面有明显的区别。",
            "Comparing the {object} between {image_ref_1} and {image_ref_2}: In the first image, the {object} "
            "appears {attr_1}, while in the second it shows {attr_2}. This difference might be due to {reason}.",
        ],
        "image_detail_expansion": [
            "关于{image_ref}中的{object}，我观察到以下详细特征：首先，从颜色上看，它呈现{color}色调；"
            "其次，从大小上看，它{size_description}；位置上，它位于画面的{position}；"
            "状态上，它看起来{state}。此外，{object}与周围环境的关系是{relation}。",
            "For the {object} in {image_ref}: Color-wise, it shows {color} tones; size-wise, it {size_description}; "
            "position-wise, it's in the {position} of the frame; condition-wise, it appears {state}.",
        ],
    }

    def __init__(
        self,
        api_url: str = "https://globalai.vip/v1/chat/completions",
        api_key: str = "sk-PcyvuAqtt0yHsP88Mga584zkJIeP7VrSC2l4QOaK0wGpSx3R",
        weak_model: str = "gpt-3.5-turbo",  # Configurable weak model
        use_weak_model: bool = True,  # Whether to actually call the model
        config: Optional[FillerConfig] = None,
        timeout: int = 30
    ):
        """
        Initialize context padder.

        Args:
            api_url: API endpoint URL
            api_key: API key
            weak_model: Name of weak model to use (e.g., gpt-3.5-turbo, gpt-4o-mini)
            use_weak_model: If True, call LLM for filler; if False, use templates
            config: Configuration for filler generation
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.api_key = api_key
        self.weak_model = weak_model
        self.use_weak_model = use_weak_model
        self.config = config or FillerConfig()
        self.timeout = timeout

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def _call_weak_model(
        self,
        topic: str,
        question: str,
        context: str = ""
    ) -> Optional[str]:
        """Call weak model to generate verbose filler response"""
        user_prompt = self.WEAK_MODEL_USER_TEMPLATE.format(
            topic=topic,
            question=question,
            context=context or "刚开始讨论，还没有具体内容。",
            min_length=self.config.min_filler_length,
            max_length=self.config.max_filler_length
        )

        payload = {
            "model": self.weak_model,
            "messages": [
                {"role": "system", "content": self.WEAK_MODEL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 800,
            "temperature": 0.9  # Higher temperature for more varied responses
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content
            else:
                logger.warning(f"Weak model call failed: {response.status_code}")
                return None

        except Exception as e:
            logger.warning(f"Weak model exception: {e}")
            return None

    def _generate_template_response(self, topic: str) -> str:
        """Generate verbose response using templates (fallback)"""
        # More elaborate template responses
        response_parts = []

        opening_phrases = [
            "让我仔细观察一下这个问题。",
            "这是一个很好的观察点。",
            "关于这个方面，我有一些想法想分享。",
            "I'll take a careful look at this.",
            "That's a great point to consider.",
            "Regarding this aspect, I have some thoughts to share.",
        ]
        response_parts.append(random.choice(opening_phrases))

        # Main content based on topic
        main_content = {
            "background_details": [
                "从背景来看，我可以观察到多层次的视觉元素。首先是色调方面，整体呈现出一种特定的氛围。光线从某个方向照射进来，在物体上形成明暗对比。空间布局上，前景、中景和背景之间有明显的层次感。",
                "The background reveals multiple layers of visual elements. First, in terms of color tones, there's a specific atmosphere being conveyed. Light enters from a certain direction, creating contrast on objects. Spatially, there's a clear sense of depth between foreground, middle ground, and background.",
            ],
            "scene_description": [
                "这个场景整体给人一种特定的感觉。从构图角度看，主体元素被安排在视觉焦点位置。色彩搭配上，有冷暖色调的对比或和谐。光影效果营造出特定的时间感和空间感。整体来说，这是一个具有叙事性的画面。",
                "This scene conveys a particular feeling overall. From a compositional standpoint, the main elements are arranged at focal points. In terms of color coordination, there's either contrast or harmony between warm and cool tones. The lighting effects create a specific sense of time and space.",
            ],
            "tangential_observation": [
                "有趣的是，我注意到了一些初看可能会忽略的细节。这些小元素虽然不是主体，但它们为整体画面增添了丰富性。它们的存在让场景更加真实和完整。",
                "Interestingly, I've noticed some details that might be overlooked at first glance. While these small elements aren't the main subject, they add richness to the overall image. Their presence makes the scene more authentic and complete.",
            ],
        }

        topic_content = main_content.get(topic, main_content["tangential_observation"])
        response_parts.append(random.choice(topic_content))

        # Add filler sentences
        fillers = [
            "这种观察对于理解整体画面很重要。",
            "当然，这只是我个人的理解，可能还有其他角度。",
            "如果仔细看的话，还能发现更多细节。",
            "This observation is important for understanding the overall picture.",
            "Of course, this is just my personal interpretation - there might be other perspectives.",
            "Looking more carefully, one can discover even more details.",
        ]
        response_parts.append(random.choice(fillers))

        # Closing
        closings = [
            "总的来说，这是一个值得细细品味的画面。",
            "以上就是我的主要观察。",
            "Overall, this is an image worth examining carefully.",
            "Those are my main observations.",
        ]
        response_parts.append(random.choice(closings))

        return " ".join(response_parts)

    def generate_multi_image_filler_turn(
        self,
        turn_context: Dict[str, Any],
        topic: Optional[str] = None,
        image_refs: Optional[List[str]] = None,
        objects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a filler turn specifically for multi-image conversations.

        Args:
            turn_context: Context about current conversation
            topic: Specific topic for filler (random if None)
            image_refs: List of image references (e.g., ["Image 1", "Image 2"])
            objects: List of objects mentioned in the images

        Returns:
            Dict with "user_message", "model_response", and metadata
        """
        # Default image refs and objects
        image_refs = image_refs or ["图片1", "图片2", "Image 1", "Image 2"]
        objects = objects or ["物体", "object", "人", "person", "东西", "thing"]

        # Select topic
        multi_image_topics = list(self.MULTI_IMAGE_FILLER_USER_TEMPLATES.keys())
        topic = topic if topic in multi_image_topics else random.choice(multi_image_topics)

        # Select user message template
        templates = self.MULTI_IMAGE_FILLER_USER_TEMPLATES.get(topic, [])
        if not templates:
            templates = self.MULTI_IMAGE_FILLER_USER_TEMPLATES["image_detail_expansion"]

        template = random.choice(templates)

        # Fill in template parameters
        try:
            user_message = template.format(
                image_ref=random.choice(image_refs),
                image_ref_1=image_refs[0] if len(image_refs) > 0 else "图片1",
                image_ref_2=image_refs[1] if len(image_refs) > 1 else "图片2",
                object=random.choice(objects),
                related_object=random.choice(objects),
                attribute=random.choice(["颜色", "大小", "位置", "color", "size", "position"])
            )
        except (KeyError, IndexError):
            user_message = f"关于{random.choice(image_refs)}中的{random.choice(objects)}，能详细说说吗？"

        # Generate verbose response
        model_response = None

        if self.use_weak_model:
            context_summary = turn_context.get("summary", "")
            model_response = self._call_weak_model(
                topic=topic,
                question=user_message,
                context=context_summary
            )

        if not model_response:
            # Fallback to template
            model_response = self._generate_multi_image_template_response(topic, image_refs, objects)

        return {
            "user_message": user_message,
            "model_response": model_response,
            "topic": topic,
            "is_filler": True,
            "is_multi_image": True,
            "response_length": len(model_response)
        }

    def _generate_multi_image_template_response(
        self,
        topic: str,
        image_refs: List[str],
        objects: List[str]
    ) -> str:
        """Generate verbose response for multi-image scenarios using templates"""
        response_parts = []

        # Opening
        openings = [
            "让我仔细分析一下这个问题。",
            "这是一个很好的观察角度。",
            "关于多张图片的比较，我有以下观察：",
            "Let me analyze this carefully.",
            "That's a great observation to make.",
        ]
        response_parts.append(random.choice(openings))

        # Main content based on topic
        obj = random.choice(objects)
        img1 = image_refs[0] if image_refs else "图片1"
        img2 = image_refs[1] if len(image_refs) > 1 else "图片2"

        if topic == "cross_image_comparison":
            content = [
                f"比较{img1}和{img2}中的{obj}，可以发现一些有趣的差异。"
                f"在{img1}中，{obj}呈现出某种特征，而在{img2}中则有所不同。"
                f"这种差异可能与拍摄角度、光线条件或物体本身的变化有关。",
                f"Looking at the {obj} across both images, there are notable differences. "
                f"The {obj} in the first image shows certain characteristics that differ from the second.",
            ]
        elif topic == "object_tracking":
            content = [
                f"回顾我们讨论过的图片，{obj}在每张图中都有不同的呈现方式。"
                f"在{img1}中，我们看到{obj}具有一些特点；"
                f"而在{img2}中，{obj}的表现又有所不同。这种跨图追踪有助于理解整体情况。",
                f"Tracking the {obj} across images: In {img1}, the {obj} shows certain features; "
                f"in {img2}, we see different aspects.",
            ]
        else:
            content = [
                f"关于{img1}中的{obj}，我观察到了多个方面的特征。"
                f"从视觉上看，它与{img2}中的相关元素有一定的联系。"
                f"这种观察可以帮助我们更好地理解整体画面。",
                f"Regarding the {obj} in {img1}, I've noticed multiple aspects. "
                f"Visually, it connects with elements in {img2} in interesting ways.",
            ]

        response_parts.append(random.choice(content))

        # Filler sentences
        fillers = [
            "这种对比分析对于理解多图场景很有帮助。",
            "当然，这只是基于目前观察的初步判断。",
            "如果需要更详细的分析，可以进一步讨论。",
            "This comparative analysis helps understand multi-image scenarios.",
            "Of course, this is just an initial observation based on what we've seen.",
        ]
        response_parts.append(random.choice(fillers))

        return " ".join(response_parts)

    def generate_filler_turn(
        self,
        turn_context: Dict[str, Any],
        topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a filler turn (question + verbose response).

        Args:
            turn_context: Context about current conversation
            topic: Specific topic for filler (random if None)

        Returns:
            Dict with "user_message", "model_response", and metadata
        """
        topic = topic or random.choice(self.config.topics)

        # Select user message template
        templates = self.FILLER_USER_TEMPLATES.get(
            topic,
            self.FILLER_USER_TEMPLATES["tangential_observation"]
        )
        user_message = random.choice(templates)

        # Generate verbose response
        model_response = None

        if self.use_weak_model:
            # Try to get response from weak model
            context_summary = turn_context.get("summary", "")
            model_response = self._call_weak_model(
                topic=topic,
                question=user_message,
                context=context_summary
            )

        if not model_response:
            # Fallback to template
            model_response = self._generate_template_response(topic)

        # Ensure response is long enough
        if len(model_response) < self.config.min_filler_length:
            # Pad with additional content
            padding = [
                "让我补充一些更多的细节。" + self._generate_template_response(topic),
                "Additionally, I'd like to add more details. " + self._generate_template_response(topic),
            ]
            model_response = model_response + " " + random.choice(padding)

        return {
            "user_message": user_message,
            "model_response": model_response,
            "topic": topic,
            "is_filler": True,
            "response_length": len(model_response)
        }

    def generate_padding_sequence(
        self,
        main_turns: List[Dict[str, Any]],
        target_length: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Generate a padded conversation sequence.

        Args:
            main_turns: The actual important turns
            target_length: Desired total turn count

        Returns:
            Padded turn sequence with fillers inserted
        """
        filler_count = max(0, target_length - len(main_turns))
        filler_count = min(filler_count, self.config.max_filler_turns * 2)

        if filler_count == 0:
            return main_turns

        # Generate filler turns
        filler_turns = []
        context_summary = ""

        for i in range(filler_count):
            topic = random.choice(self.config.topics)
            context = {
                "turn_index": i,
                "summary": context_summary
            }
            filler = self.generate_filler_turn(context, topic)
            filler_turns.append({
                "type": "filler",
                **filler
            })

            # Update context summary
            context_summary += f" Turn {i+1}: discussed {topic}."

        # Distribute fillers according to config
        return self._insert_fillers(main_turns, filler_turns)

    def _insert_fillers(
        self,
        main_turns: List[Dict[str, Any]],
        fillers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Insert filler turns into main sequence"""
        if not fillers:
            return main_turns

        result = []
        position = self.config.filler_position

        if position == "early":
            # Put most fillers at the beginning
            result.extend(fillers[:len(fillers)//2])
            result.extend(main_turns)
            result.extend(fillers[len(fillers)//2:])

        elif position == "middle":
            # Put fillers in the middle
            mid = len(main_turns) // 2
            result.extend(main_turns[:mid])
            result.extend(fillers)
            result.extend(main_turns[mid:])

        elif position == "late":
            # Put fillers near the end (but not after final question)
            if len(main_turns) > 1:
                result.extend(main_turns[:-1])
                result.extend(fillers)
                result.append(main_turns[-1])
            else:
                result.extend(fillers)
                result.extend(main_turns)

        else:  # "distributed"
            # Distribute evenly throughout
            main_idx = 0
            filler_idx = 0

            # Calculate insertion interval
            if len(main_turns) > 0:
                interval = max(1, len(main_turns) // (len(fillers) + 1))
            else:
                interval = 1

            turn_count = 0
            while main_idx < len(main_turns) or filler_idx < len(fillers):
                # Add a main turn
                if main_idx < len(main_turns):
                    result.append(main_turns[main_idx])
                    main_idx += 1
                    turn_count += 1

                # Add filler after every N main turns
                if turn_count > 0 and turn_count % interval == 0 and filler_idx < len(fillers):
                    result.append(fillers[filler_idx])
                    filler_idx += 1

            # Add remaining fillers at the end (but before last main turn if possible)
            while filler_idx < len(fillers):
                if len(result) > 1:
                    result.insert(-1, fillers[filler_idx])
                else:
                    result.append(fillers[filler_idx])
                filler_idx += 1

        return result

    def inject_filler_into_conversation(
        self,
        conversation: List[Dict[str, str]],
        num_fillers: int = 3,
        position: str = "distributed"
    ) -> List[Dict[str, str]]:
        """
        Inject filler turns directly into an existing conversation.

        Args:
            conversation: Existing conversation as list of {"role": "...", "content": "..."}
            num_fillers: Number of filler exchanges to inject
            position: Where to inject ("early", "middle", "late", "distributed")

        Returns:
            Conversation with filler turns injected
        """
        result = list(conversation)  # Copy

        for i in range(num_fillers):
            filler = self.generate_filler_turn({"turn_index": i})

            # Create conversation turns
            user_turn = {"role": "user", "content": filler["user_message"]}
            assistant_turn = {"role": "assistant", "content": filler["model_response"]}

            # Calculate insertion position
            if position == "early":
                insert_idx = min(2 + i * 2, len(result))
            elif position == "middle":
                insert_idx = len(result) // 2
            elif position == "late":
                insert_idx = max(0, len(result) - 2)
            else:  # distributed
                interval = len(result) // (num_fillers + 1)
                insert_idx = min((i + 1) * interval, len(result))

            result.insert(insert_idx, user_turn)
            result.insert(insert_idx + 1, assistant_turn)

        return result


def create_long_context_conversation(
    core_turns: List[Dict[str, Any]],
    target_turns: int = 30,
    api_url: str = "https://globalai.vip/v1/chat/completions",
    api_key: str = "sk-PcyvuAqtt0yHsP88Mga584zkJIeP7VrSC2l4QOaK0wGpSx3R",
    weak_model: str = "gpt-3.5-turbo",
    use_weak_model: bool = True
) -> List[Dict[str, Any]]:
    """
    Convenience function to create a long-context conversation.

    Args:
        core_turns: Essential conversation turns
        target_turns: Desired conversation length
        api_url: API endpoint
        api_key: API key
        weak_model: Weak model name for filler generation
        use_weak_model: Whether to use LLM for filler

    Returns:
        Padded conversation with filler turns
    """
    padder = ContextPadder(
        api_url=api_url,
        api_key=api_key,
        weak_model=weak_model,
        use_weak_model=use_weak_model,
        config=FillerConfig(
            min_filler_turns=3,
            max_filler_turns=15,
            min_filler_length=150,
            max_filler_length=500,
            filler_position="distributed"
        )
    )

    return padder.generate_padding_sequence(core_turns, target_turns)


# Example usage and test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Context Padder...")

    # Test with templates only (no API calls)
    padder = ContextPadder(use_weak_model=False)

    # Test single filler generation
    filler = padder.generate_filler_turn({"summary": "Discussing people in image"})
    print("\n--- Single Filler (Template) ---")
    print(f"User: {filler['user_message']}")
    print(f"Model ({filler['response_length']} chars): {filler['model_response'][:200]}...")

    # Test padding sequence
    main_turns = [
        {"type": "main", "content": "Look at image 1"},
        {"type": "main", "content": "Now compare with image 2"},
        {"type": "main", "content": "Which has more people?"},
    ]

    padded = padder.generate_padding_sequence(main_turns, target_length=10)
    print(f"\n--- Padded Sequence ({len(padded)} turns) ---")
    for i, turn in enumerate(padded):
        turn_type = turn.get("type", "unknown")
        length = turn.get("response_length", 0)
        print(f"Turn {i+1}: [{turn_type}] {f'({length} chars)' if length else ''}")
