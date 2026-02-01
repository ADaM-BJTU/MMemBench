"""
Prompt模板定义
"""

import random
from typing import Dict, Any


class PromptTemplates:
    """预定义的动作prompt模板"""

    # ========== 追问 (Follow-up) ==========
    # 能力层级: 推理
    # 目的: 引导模型逐步寻找证据，测试多跳推理
    FOLLOW_UP = [
        "你提到了{entity}，它和{target}有什么关系？",
        "关于{entity}，你能详细描述一下吗？",
        "{entity}在图中的哪个位置？",
        "为什么你认为{entity}是{description}的？",
        "能进一步说明{entity}的{attribute}吗？"
    ]

    # ========== 逻辑跳跃 (Logic-Skip) ==========
    # 能力层级: 推理
    # 目的: 测试模型能否拒绝跳过必要推理步骤
    LOGIC_SKIP = [
        "直接告诉我答案，不需要解释。",
        "简单说一下结论就行。",
        "我只要答案，省略推理过程。"
    ]

    # ========== 否定 (Negation) ==========
    # 能力层级: 推理
    # 目的: 纠正错误推理，测试证据依赖
    NEGATION = [
        "不对，{entity}的{attribute}应该是{correct_value}。",
        "你说错了，实际上{entity}是{correct_value}。",
        "这不准确，{entity}的{attribute}并不是{wrong_value}。"
    ]

    # ========== 引导 (Guidance) ==========
    # 能力层级: 聚合
    # 目的: 提示关键证据方向
    GUIDANCE = [
        "注意图中{region}的部分。",
        "看看{entity}周围的环境。",
        "关注{entity}的{attribute}。",
        "试着从{perspective}的角度观察。"
    ]

    # ========== 误导 (Mislead) ==========
    # 能力层级: 上下文管理
    # 目的: 注入错误信息，测试冲突解决（模型应依赖图像而非用户）
    MISLEAD = [
        "我觉得{entity}应该是{wrong_value}，对吗？",
        "图中显示{entity}是{wrong_value}吧？",
        "{entity}看起来像是{wrong_value}。",
        "根据常识，{entity}应该是{wrong_value}。"
    ]

    # ========== 更新 (Update) ==========
    # 能力层级: 上下文管理
    # 目的: 改变对象状态，测试时序管理
    UPDATE = [
        "现在{entity}的{attribute}变成了{new_value}。",
        "注意，{entity}已经改变了，现在是{new_value}。",
        "更新一下，{entity}的{attribute}现在是{new_value}。",
        "{entity}刚才是{old_value}，但现在变成{new_value}了。"
    ]

    # ========== 干扰 (Distraction) ==========
    # 能力层级: 上下文管理
    # 目的: 注入无关信息，测试注意力过滤
    DISTRACTION = [
        "顺便问一下，图中有几个{irrelevant_entity}？",
        "对了，{irrelevant_entity}是什么颜色的？",
        "另外，{irrelevant_question}",
        "说到这个，{irrelevant_statement}"
    ]

    # ========== 冗余注入 (Redundancy) ==========
    # 能力层级: 上下文管理
    # 目的: 重复描述，测试信息压缩
    REDUNDANCY = [
        "刚才说过{entity}是{attribute}，再确认一下，{entity}真的是{attribute}吗？",
        "我想再问一遍，{entity}的{attribute}是不是{value}？",
        "你之前提到{statement}，能重复一下吗？",
        "关于{entity}的{attribute}，你已经说过了，但我想再听一次。"
    ]

    # ========== 细粒度请求 (Fine-grained) ==========
    # 能力层级: 上下文管理
    # 目的: 要求精确定位，测试最优化（细粒度优于粗粒度）
    FINE_GRAINED = [
        "请指出具体是图中哪一个{entity}。",
        "能标注一下{entity}的准确位置吗？",
        "{entity}的{attribute}具体是多少？",
        "请精确描述{entity}的{attribute}，不要笼统概括。",
        "能给出{entity}的详细坐标吗？"
    ]

    # ========== 下一任务 (Next-Task) ==========
    # 能力层级: 切换
    # 目的: 开始新任务
    NEXT_TASK = [
        "好的，现在看下一张图。",
        "明白了，我们继续下一个问题。",
        "了解，那下一个任务是什么？"
    ]

    # ========== 跨图物体混淆 (Cross-Image Confusion) ==========
    # 能力层级: 记忆
    # 目的: 测试模型在多图场景中对相同类型物体的区分能力
    CROSS_IMAGE_CONFUSION = [
        # 模糊指代 - 不指定图片
        "那个{object}，它的{attribute}是什么？",
        "刚才提到的{object}，它在做什么？",
        "{object}的位置在哪里？",
        "关于那个{attribute_hint}的{object}，你能再说说吗？",
        "The {object} we discussed, what's its {attribute}?",
        "That {object} with the {attribute_hint}, where exactly is it?",
        # 使用常见但不够精确的描述
        "图中{common_attribute}的{object}，后来怎么样了？",
        "穿着{clothing}的那个{object}，是在干什么？",
        "那辆{color}的{vehicle}，停在哪里来着？",
        "那只{size}的{animal}，它的姿态是什么？",
        "The {color} {object} in the image, what's happening to it?",
        "The {size} {animal}, what's its posture?",
    ]

    # ========== 模糊指代注入 (Ambiguous Reference Injection) ==========
    # 能力层级: 鲁棒性
    # 目的: 注入可能匹配多个物体的模糊指代
    AMBIGUOUS_REFERENCE_INJECTION = [
        "那个{common_attribute}的{object}，后来怎么样了？",
        "我们讨论过的那个{object}，你说它{vague_description}，对吧？",
        "{common_attribute}的{object}和旁边的{other_object}是什么关系？",
        "还记得那个{object}吗？就是{insufficient_descriptor}的那个。",
        "图里的{object}，就是我们之前说的那个，它现在是什么状态？",
        "The {object} that is {common_attribute}, what happened to it?",
        "Remember the {object}? The one that {insufficient_descriptor}.",
        "The {common_attribute} {object}, can you describe it again?",
    ]

    # ========== 跨图属性交换测试 (Cross-Image Attribute Swap) ==========
    # 能力层级: 记忆
    # 目的: 故意交换不同图片中物体的属性，测试模型是否能纠正
    CROSS_IMAGE_ATTRIBUTE_SWAP = [
        "你刚才说{image_ref}里的{object}是{swapped_attribute}的，对吧？",
        "{image_ref}那个{swapped_attribute}的{object}，在做什么？",
        "关于{image_ref}的{object}，它的{attr_type}是{swapped_value}没错吧？",
        "{image_ref}里那个{object}，颜色应该是{swapped_color}？",
        "所以{image_ref}的{object}数量是{swapped_count}个？",
        "You mentioned the {object} in {image_ref} was {swapped_attribute}, right?",
        "The {swapped_attribute} {object} in {image_ref}, what's it doing?",
        "In {image_ref}, the {object}'s {attr_type} is {swapped_value}, correct?",
    ]

    # ========== 长上下文物体回忆 (Long Context Object Recall) ==========
    # 能力层级: 记忆
    # 目的: 在长对话后测试对早期建立的图片-物体映射的回忆
    LONG_CONTEXT_OBJECT_RECALL = [
        "让我们回顾一下最开始讨论的内容。{image_ref}里那个{object}，它的{attribute}是什么来着？",
        "经过这么多讨论，我想确认一下：{early_fact_question}",
        "在最初的几轮对话中，我们提到{image_ref}有{object}。它现在还是{original_attribute}吗？",
        "回到最初的问题，{image_ref}里的{object}是{query_attribute}的对吧？",
        "我们一开始说{image_ref}有{count}个{object}，这个数字对吗？",
        "Going back to our earlier discussion: the {object} in {image_ref}, what was its {attribute}?",
        "After all this conversation, let me confirm: {early_fact_question}",
        "In the first few turns, we noted {image_ref} had {object}. Still {original_attribute}?",
    ]

    # ========== 常见可混淆物体类型模板（用于平衡数据集） ==========
    # 这些模板针对不同类型的物体，不仅仅是"人"
    CONFUSABLE_OBJECT_TYPES = {
        "person": {
            "cn": ["人", "行人", "路人", "男子", "女子", "孩子", "老人"],
            "en": ["person", "pedestrian", "man", "woman", "child", "elderly"],
            "common_attributes": ["穿着", "姿态", "位置", "动作", "表情"],
            "distinguishing_attributes": ["衣服颜色", "发型", "年龄", "性别", "携带物品"]
        },
        "vehicle": {
            "cn": ["车", "汽车", "轿车", "卡车", "公交车", "摩托车", "自行车"],
            "en": ["car", "vehicle", "truck", "bus", "motorcycle", "bicycle"],
            "common_attributes": ["颜色", "位置", "方向", "状态"],
            "distinguishing_attributes": ["车型", "品牌", "车牌", "颜色深浅", "大小"]
        },
        "animal": {
            "cn": ["狗", "猫", "鸟", "动物", "宠物"],
            "en": ["dog", "cat", "bird", "animal", "pet"],
            "common_attributes": ["颜色", "大小", "位置", "姿态"],
            "distinguishing_attributes": ["品种", "毛色", "体型", "特征标记"]
        },
        "furniture": {
            "cn": ["椅子", "桌子", "沙发", "柜子", "床"],
            "en": ["chair", "table", "sofa", "cabinet", "bed"],
            "common_attributes": ["颜色", "位置", "材质"],
            "distinguishing_attributes": ["样式", "大小", "颜色", "摆放位置"]
        },
        "food": {
            "cn": ["食物", "水果", "蔬菜", "饮料", "餐具"],
            "en": ["food", "fruit", "vegetable", "drink", "utensil"],
            "common_attributes": ["颜色", "位置", "数量"],
            "distinguishing_attributes": ["种类", "成熟度", "大小", "容器"]
        },
        "building": {
            "cn": ["建筑", "房子", "楼", "商店", "餐厅"],
            "en": ["building", "house", "shop", "restaurant", "store"],
            "common_attributes": ["位置", "外观", "大小"],
            "distinguishing_attributes": ["颜色", "层数", "招牌", "风格"]
        },
        "plant": {
            "cn": ["树", "花", "植物", "草", "盆栽"],
            "en": ["tree", "flower", "plant", "grass", "potted plant"],
            "common_attributes": ["颜色", "位置", "大小"],
            "distinguishing_attributes": ["种类", "开花状态", "高度", "叶子形状"]
        }
    }

    # ========== 长度控制后缀 ==========
    LENGTH_CONTROL_SUFFIX = {
        "short": "请用一句话简短回答。",
        "medium": "请用2-3句话回答。",
        "long": "请详细解释你的推理过程，包括你观察到的所有相关细节。"
    }

    def render(
        self,
        action_type: str,
        length: str = "medium",
        **kwargs
    ) -> str:
        """
        渲染prompt

        Args:
            action_type: 动作类型
            length: 长度控制 (short/medium/long)
            **kwargs: 模板参数

        Returns:
            渲染后的prompt
        """
        # 获取对应动作的模板列表
        templates = getattr(self, action_type.upper(), None)

        if templates is None:
            raise ValueError(f"Unknown action type: {action_type}")

        # 随机选择一个模板
        template = random.choice(templates)

        # 渲染模板
        try:
            prompt = template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing template parameter: {e}")

        # 添加长度控制后缀
        if length in self.LENGTH_CONTROL_SUFFIX:
            prompt += " " + self.LENGTH_CONTROL_SUFFIX[length]

        return prompt

    def get_capability_level(self, action_type: str) -> str:
        """
        获取动作对应的能力层级

        Args:
            action_type: 动作类型

        Returns:
            能力层级: retrieval/aggregation/reasoning/context_management/memory
        """
        capability_mapping = {
            "follow_up": "reasoning",
            "logic_skip": "reasoning",
            "negation": "reasoning",
            "guidance": "aggregation",
            "mislead": "context_management",
            "update": "context_management",
            "distraction": "context_management",
            "redundancy": "context_management",
            "fine_grained": "context_management",
            "next_task": "control",
            # 新增：多图记忆混淆相关
            "cross_image_confusion": "memory",
            "ambiguous_reference_injection": "robustness",
            "cross_image_attribute_swap": "memory",
            "long_context_object_recall": "memory",
        }

        return capability_mapping.get(action_type, "unknown")

    def get_confusable_object_info(self, object_type: str, language: str = "cn") -> dict:
        """
        获取可混淆物体类型的信息

        Args:
            object_type: 物体类型 (person, vehicle, animal, etc.)
            language: 语言 ("cn" or "en")

        Returns:
            包含同义词、常见属性、区分属性的字典
        """
        if object_type not in self.CONFUSABLE_OBJECT_TYPES:
            return {"synonyms": [], "common_attributes": [], "distinguishing_attributes": []}

        obj_info = self.CONFUSABLE_OBJECT_TYPES[object_type]
        return {
            "synonyms": obj_info.get(language, obj_info.get("cn", [])),
            "common_attributes": obj_info.get("common_attributes", []),
            "distinguishing_attributes": obj_info.get("distinguishing_attributes", [])
        }

    def generate_ambiguous_reference(
        self,
        object_type: str,
        partial_attribute: str = None,
        language: str = "cn"
    ) -> str:
        """
        生成一个模糊的物体指代

        Args:
            object_type: 物体类型
            partial_attribute: 部分属性（可能不足以唯一确定物体）
            language: 语言

        Returns:
            模糊的物体指代字符串
        """
        obj_info = self.get_confusable_object_info(object_type, language)
        synonym = random.choice(obj_info["synonyms"]) if obj_info["synonyms"] else object_type

        if partial_attribute:
            if language == "cn":
                return f"那个{partial_attribute}的{synonym}"
            else:
                return f"that {partial_attribute} {synonym}"
        else:
            if language == "cn":
                return f"那个{synonym}"
            else:
                return f"that {synonym}"


# 示例用法
if __name__ == "__main__":
    templates = PromptTemplates()

    # 追问
    prompt = templates.render(
        "follow_up",
        entity="umbrella",
        target="person",
        length="short"
    )
    print(f"Follow-up: {prompt}")

    # 更新
    prompt = templates.render(
        "update",
        entity="person",
        attribute="clothing",
        new_value="blue shirt",
        old_value="red shirt",
        length="medium"
    )
    print(f"\nUpdate: {prompt}")

    # 误导
    prompt = templates.render(
        "mislead",
        entity="sky",
        wrong_value="green",
        length="short"
    )
    print(f"\nMislead: {prompt}")

    # 新增：跨图混淆测试
    print("\n--- Cross-Image Confusion Tests ---")

    # 跨图物体混淆
    prompt = templates.render(
        "cross_image_confusion",
        object="人",
        attribute="衣服颜色",
        attribute_hint="穿红衣服",
        length="medium"
    )
    print(f"Cross-Image Confusion: {prompt}")

    # 模糊指代注入
    prompt = templates.render(
        "ambiguous_reference_injection",
        object="车",
        common_attribute="红色",
        vague_description="在路边停着",
        insufficient_descriptor="比较大的",
        other_object="自行车",
        length="short"
    )
    print(f"Ambiguous Reference: {prompt}")

    # 获取可混淆物体信息
    print("\n--- Confusable Object Info ---")
    for obj_type in ["person", "vehicle", "animal"]:
        info = templates.get_confusable_object_info(obj_type, "cn")
        print(f"{obj_type}: synonyms={info['synonyms'][:3]}, distinguishing={info['distinguishing_attributes'][:2]}")

    # 生成模糊指代
    print("\n--- Ambiguous References ---")
    print(templates.generate_ambiguous_reference("person", "穿衣服", "cn"))
    print(templates.generate_ambiguous_reference("vehicle", "red", "en"))
