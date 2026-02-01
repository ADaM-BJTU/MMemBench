"""
查询生成器 - 根据动作类型和entities生成用户查询

设计理念:
1. 使用提取好的entities填充模板
2. 与entity提取和动作选择解耦
3. 支持灵活的模板扩展
"""

import random
from typing import Dict, List, Any, Optional

try:
    from .prompt_templates import PromptTemplates
except ImportError:
    # 当作为独立脚本运行时
    from prompt_templates import PromptTemplates


class QueryGenerator:
    """
    查询生成器

    职责:
    1. 根据动作类型选择合适的模板
    2. 从entities中选择合适的实体填充模板
    3. 添加长度控制等修饰
    """

    def __init__(self, templates: Optional[PromptTemplates] = None):
        """
        Args:
            templates: Prompt模板实例，None则创建默认实例
        """
        self.templates = templates or PromptTemplates()

    def generate(
        self,
        action_type: str,
        entities: Dict[str, Any],
        length: str = "medium",
        vlm_response: str = "",
        history: Optional[List[Dict]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        生成用户查询

        Args:
            action_type: 动作类型
            entities: 提取的entities字典
            length: 长度控制 (short/medium/long)
            vlm_response: VLM的原始响应（用于某些动作）
            history: 对话历史
            context: 额外上下文

        Returns:
            生成的query字符串
        """
        history = history or []
        context = context or {}

        # 根据动作类型准备模板参数
        template_params = self._prepare_template_params(
            action_type=action_type,
            entities=entities,
            vlm_response=vlm_response,
            history=history,
            context=context
        )

        # 使用模板生成query
        try:
            query = self.templates.render(
                action_type=action_type,
                length=length,
                **template_params
            )
        except (KeyError, ValueError) as e:
            # 如果模板参数缺失，生成一个通用query
            print(f"Warning: Failed to render template for {action_type}: {e}")
            query = self._generate_fallback_query(action_type, length)

        return query

    def _prepare_template_params(
        self,
        action_type: str,
        entities: Dict[str, Any],
        vlm_response: str,
        history: List[Dict],
        context: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        根据动作类型准备模板参数

        这里定义了每个动作类型需要哪些entities
        """
        params = {}

        # 1. Follow-up: 需要entity、target、attribute
        if action_type == "follow_up":
            params["entity"] = self._get_random_entity(entities, "objects", "这个")
            params["target"] = self._get_random_entity(entities, "objects", "目标", exclude=[params["entity"]])
            params["attribute"] = self._get_random_attribute(entities, params["entity"])
            params["description"] = self._get_random_attribute_value(entities, params["entity"])

        # 2. Guidance: 需要region、entity、aspect
        elif action_type == "guidance":
            params["region"] = self._get_random_entity(entities, "regions", "左上角")
            params["entity"] = self._get_random_entity(entities, "objects", "物体")
            params["aspect"] = self._get_random_attribute(entities, params["entity"], default="细节")
            params["perspective"] = random.choice(["整体", "局部", "细节", "背景"])

        # 3. Negation: 需要entity、attribute、correct_value、wrong_value
        elif action_type == "negation":
            params["entity"] = self._get_random_entity(entities, "objects", "这个")
            params["attribute"] = self._get_random_attribute(entities, params["entity"])

            # 从entities中获取正确值，生成错误值
            correct_value = self._get_attribute_value(entities, params["entity"], params["attribute"])
            params["correct_value"] = correct_value or "正确的值"
            params["wrong_value"] = self._generate_wrong_value(correct_value)

        # 4. Mislead: 需要entity、wrong_value
        elif action_type == "mislead":
            params["entity"] = self._get_random_entity(entities, "objects", "这个")

            # 生成一个明显错误的值
            correct_value = self._get_attribute_value(entities, params["entity"], "color")
            params["wrong_value"] = self._generate_wrong_value(correct_value, misleading=True)

        # 5. Update: 需要entity、attribute、new_value、old_value
        elif action_type == "update":
            params["entity"] = self._get_random_entity(entities, "objects", "物体")
            params["attribute"] = self._get_random_attribute(entities, params["entity"])

            old_value = self._get_attribute_value(entities, params["entity"], params["attribute"])
            params["old_value"] = old_value or "原来的值"
            params["new_value"] = self._generate_new_value(old_value)

        # 6. Distraction: 需要irrelevant_entity、irrelevant_question
        elif action_type == "distraction":
            # 生成与当前entities无关的内容
            params["irrelevant_entity"] = random.choice(["云", "树", "建筑", "背景", "其他物体"])
            params["irrelevant_question"] = random.choice([
                "天气怎么样",
                "时间是什么时候",
                "这是在哪里拍的",
                "图片是什么时候拍的"
            ])
            params["irrelevant_statement"] = random.choice([
                "这让我想起了另一个场景",
                "这很有趣",
                "我想知道其他细节"
            ])

        # 7. Redundancy: 需要entity、attribute、statement
        elif action_type == "redundancy":
            params["entity"] = self._get_random_entity(entities, "objects", "刚才提到的")
            params["attribute"] = self._get_random_attribute(entities, params["entity"])
            params["value"] = self._get_attribute_value(entities, params["entity"], params["attribute"]) or "那个值"
            params["statement"] = vlm_response[:50] if vlm_response else "刚才的内容"

        # 8. Fine-grained: 需要entity、attribute
        elif action_type == "fine_grained":
            params["entity"] = self._get_random_entity(entities, "objects", "物体")
            params["attribute"] = random.choice(["位置", "颜色", "大小", "状态", "数量"])

        # 9. Logic-skip: 不需要特殊参数
        elif action_type == "logic_skip":
            pass  # 模板不需要参数

        # 10. Next-task: 不需要特殊参数
        elif action_type == "next_task":
            pass

        return params

    def _get_random_entity(
        self,
        entities: Dict[str, Any],
        entity_type: str,
        default: str,
        exclude: Optional[List[str]] = None
    ) -> str:
        """从entities中随机获取一个实体"""
        exclude = exclude or []

        if entity_type not in entities:
            return default

        entity_list = entities[entity_type]
        if isinstance(entity_list, list) and len(entity_list) > 0:
            # 过滤exclude
            candidates = [e for e in entity_list if e not in exclude]
            if candidates:
                return random.choice(candidates)

        return default

    def _get_random_attribute(
        self,
        entities: Dict[str, Any],
        entity: str,
        default: str = "属性"
    ) -> str:
        """获取实体的一个属性类型"""
        if "attributes" in entities and entity in entities["attributes"]:
            attrs = entities["attributes"][entity]
            if isinstance(attrs, dict) and len(attrs) > 0:
                return random.choice(list(attrs.keys()))

        # 默认属性类型
        return random.choice(["颜色", "大小", "状态", "位置"])

    def _get_random_attribute_value(
        self,
        entities: Dict[str, Any],
        entity: str,
        default: str = "某个特征"
    ) -> str:
        """获取实体的一个属性值"""
        if "attributes" in entities and entity in entities["attributes"]:
            attrs = entities["attributes"][entity]
            if isinstance(attrs, dict) and len(attrs) > 0:
                # 随机选择一个属性类型
                attr_type = random.choice(list(attrs.keys()))
                values = attrs[attr_type]
                if isinstance(values, list) and len(values) > 0:
                    return values[0]

        return default

    def _get_attribute_value(
        self,
        entities: Dict[str, Any],
        entity: str,
        attribute: str
    ) -> Optional[str]:
        """获取指定实体的指定属性值"""
        if "attributes" in entities and entity in entities["attributes"]:
            attrs = entities["attributes"][entity]
            if isinstance(attrs, dict) and attribute in attrs:
                values = attrs[attribute]
                if isinstance(values, list) and len(values) > 0:
                    return values[0]
        return None

    def _generate_wrong_value(
        self,
        correct_value: Optional[str],
        misleading: bool = False
    ) -> str:
        """生成一个错误的值（用于negation和mislead）"""
        # 颜色替换表
        color_map = {
            "红色": "蓝色", "red": "blue",
            "蓝色": "绿色", "blue": "green",
            "绿色": "黄色", "green": "yellow",
            "黄色": "红色", "yellow": "red",
            "黑色": "白色", "black": "white",
            "白色": "黑色", "white": "black",
        }

        if correct_value and correct_value in color_map:
            return color_map[correct_value]

        # 大小替换
        size_map = {
            "大": "小", "large": "small",
            "小": "大", "small": "large",
            "高": "矮", "tall": "short",
            "矮": "高", "short": "tall",
        }

        if correct_value and correct_value in size_map:
            return size_map[correct_value]

        # 默认错误值
        if misleading:
            return random.choice(["紫色", "粉色", "橙色", "奇怪的形状"])
        else:
            return "另一个值"

    def _generate_new_value(self, old_value: Optional[str]) -> str:
        """生成一个新值（用于update）"""
        # 复用wrong_value逻辑
        return self._generate_wrong_value(old_value)

    def _generate_fallback_query(self, action_type: str, length: str) -> str:
        """生成通用query（当模板失败时）"""
        fallback_queries = {
            "follow_up": "能详细说说吗？",
            "guidance": "注意看图中的细节。",
            "negation": "这不对。",
            "mislead": "我觉得应该是另一个值。",
            "update": "现在情况变了。",
            "distraction": "顺便问一下，还有什么？",
            "redundancy": "再说一遍。",
            "fine_grained": "能更精确一点吗？",
            "logic_skip": "直接告诉我答案。",
            "next_task": "下一个。"
        }

        query = fallback_queries.get(action_type, "继续。")

        # 添加长度控制
        suffix = self.templates.LENGTH_CONTROL_SUFFIX.get(length, "")
        if suffix:
            query += " " + suffix

        return query


# 示例用法
if __name__ == "__main__":
    from entity_extractor import EntityExtractor

    print("=" * 60)
    print("测试QueryGenerator")
    print("=" * 60)

    # 1. 准备entities
    extractor = EntityExtractor(extraction_mode="simple")
    text = "图中有一个人拿着一把黑色的伞站在街道左上角。"
    entities = extractor.extract(text)

    print(f"\n提取的entities:")
    print(f"  objects: {entities['objects']}")
    print(f"  attributes: {entities['attributes']}")
    print(f"  regions: {entities['regions']}")

    # 2. 生成不同动作的query
    generator = QueryGenerator()

    print(f"\n生成的queries:")

    for action in ["follow_up", "guidance", "negation", "update", "fine_grained"]:
        query = generator.generate(
            action_type=action,
            entities=entities,
            length="medium",
            vlm_response=text
        )
        print(f"\n  [{action}]")
        print(f"  {query}")
