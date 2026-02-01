"""
Entity提取器 - 统一从VLM响应中提取各类实体

设计理念:
1. 与动作选择解耦，一次提取，多处使用
2. 细粒度分类，方便后续扩展
3. 可独立测试和优化
4. 支持可配置的提取模式 (simple/llm)
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    统一的Entity提取器

    提取的entity类型:
    - objects: 物体/实体（人、伞、车等）
    - attributes: 属性（颜色、大小、形状等）
    - relations: 关系（在...旁边、比...大等）
    - regions: 空间区域（左上角、中心、背景等）
    - states: 状态（打开/关闭、移动/静止等）
    - values: 数值/量词（3个、50%、很多等）
    - actions: 动作（跑步、站立等）
    """

    def __init__(
        self,
        extraction_mode: str = "simple",
        llm_client: Optional["LLMClient"] = None
    ):
        """
        Args:
            extraction_mode: 提取模式
                - "simple": 简单正则匹配（快速原型，默认）
                - "ner": 使用NER模型（需要额外依赖）
                - "llm": 使用LLM提取（高质量但慢）
            llm_client: LLM 客户端（llm 模式必需）
        """
        self.extraction_mode = extraction_mode
        self.llm_client = llm_client

        # 预定义的实体词典（用于简单模式）
        self._init_entity_dict()

    def set_mode(self, mode: str, llm_client: Optional["LLMClient"] = None):
        """动态切换提取模式"""
        self.extraction_mode = mode
        if llm_client:
            self.llm_client = llm_client

    def _init_entity_dict(self):
        """初始化实体词典"""
        self.object_keywords = [
            # 人物
            '人', '男人', '女人', '小孩', '婴儿', '老人', '行人',
            # 动物
            '狗', '猫', '鸟', '马', '牛', '羊',
            # 交通工具
            '车', '汽车', '自行车', '摩托车', '公交车', '卡车', '飞机', '船',
            # 物品
            '伞', '包', '帽子', '眼镜', '手机', '电脑', '书', '椅子', '桌子',
            '杯子', '瓶子', '碗', '盘子', '勺子', '刀', '叉子',
            # 建筑/场景
            '建筑', '房子', '大楼', '桥', '塔', '门', '窗户',
            '树', '花', '草', '山', '河', '湖', '海',
            # 天气/环境
            '天空', '云', '太阳', '月亮', '星星', '雨', '雪',
            # 英文常见物体
            'person', 'man', 'woman', 'child', 'baby',
            'dog', 'cat', 'bird', 'horse',
            'car', 'bus', 'bike', 'truck',
            'umbrella', 'bag', 'hat', 'phone',
            'building', 'tree', 'sky', 'cloud'
        ]

        self.attribute_keywords = {
            'color': ['红色', '蓝色', '绿色', '黄色', '黑色', '白色', '灰色', '棕色',
                     'red', 'blue', 'green', 'yellow', 'black', 'white', 'gray'],
            'size': ['大', '小', '高', '矮', '长', '短', '宽', '窄',
                    'large', 'small', 'big', 'tall', 'short'],
            'shape': ['圆形', '方形', '三角形', '长方形',
                     'round', 'square', 'triangle'],
            'material': ['木头', '金属', '塑料', '玻璃', '布料',
                        'wood', 'metal', 'plastic', 'glass'],
            'state': ['新', '旧', '破', '干净', '脏',
                     'new', 'old', 'clean', 'dirty']
        }

        self.region_keywords = [
            '左上角', '右上角', '左下角', '右下角',
            '左边', '右边', '上方', '下方', '中间', '中心',
            '前景', '背景', '远处', '近处',
            'left', 'right', 'top', 'bottom', 'center', 'middle',
            'foreground', 'background'
        ]

        self.relation_keywords = [
            '在...旁边', '在...上面', '在...下面', '在...前面', '在...后面',
            '靠近', '远离', '之间',
            'next to', 'above', 'below', 'in front of', 'behind',
            'near', 'far from', 'between'
        ]

    def extract(
        self,
        vlm_response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        从VLM响应中提取所有类型的entity

        Args:
            vlm_response: VLM的文本响应
            context: 额外上下文信息（任务信息、图像信息等）

        Returns:
            提取的entity字典，包含:
            {
                "objects": ["人", "伞", ...],
                "attributes": {
                    "人": {"color": "红色", "state": "站立"},
                    "伞": {"color": "黑色", "state": "打开"}
                },
                "relations": [
                    {"subject": "人", "relation": "拿着", "object": "伞"}
                ],
                "regions": ["左上角", "中心"],
                "states": {"伞": "打开", "门": "关闭"},
                "values": {"人": 2, "车": 3},
                "actions": ["站立", "跑步"]
            }
        """
        context = context or {}

        if self.extraction_mode == "simple":
            return self._extract_simple(vlm_response, context)
        elif self.extraction_mode == "ner":
            return self._extract_with_ner(vlm_response, context)
        elif self.extraction_mode == "llm":
            return self._extract_with_llm(vlm_response, context)
        else:
            raise ValueError(f"Unknown extraction mode: {self.extraction_mode}")

    def _extract_simple(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        简单正则提取（快速原型）

        适合初期验证架构，后续可以无缝替换为更高级的提取方法
        """
        entities = {
            "objects": [],
            "attributes": {},
            "relations": [],
            "regions": [],
            "states": {},
            "values": {},
            "actions": []
        }

        # 1. 提取物体
        for obj in self.object_keywords:
            if obj in text:
                if obj not in entities["objects"]:
                    entities["objects"].append(obj)

        # 2. 提取属性（与物体关联）
        for obj in entities["objects"]:
            obj_attrs = {}

            # 查找该物体附近的属性词
            for attr_type, attr_list in self.attribute_keywords.items():
                for attr in attr_list:
                    # 简单匹配：属性词在物体前后一定范围内
                    pattern = f"({attr}.*?{obj}|{obj}.*?{attr})"
                    if re.search(pattern, text):
                        if attr_type not in obj_attrs:
                            obj_attrs[attr_type] = []
                        obj_attrs[attr_type].append(attr)

            if obj_attrs:
                entities["attributes"][obj] = obj_attrs

        # 3. 提取区域
        for region in self.region_keywords:
            if region in text:
                if region not in entities["regions"]:
                    entities["regions"].append(region)

        # 4. 提取数值（如"3个人"、"两辆车"）
        # 中文数字
        number_patterns = [
            r'(\d+|一|二|三|四|五|六|七|八|九|十)个?(.+?)(?:[，。；！\s]|$)',
            r'有(\d+|一|二|三|四|五|六|七|八|九|十)(.+?)(?:[，。；！\s]|$)',
        ]

        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            for num, obj in matches:
                obj = obj.strip()
                if obj in entities["objects"]:
                    entities["values"][obj] = self._parse_number(num)

        # 5. 提取关系（简化版）
        relation_patterns = [
            r'(.+?)(拿着|握着|穿着|戴着|站在|坐在)(.+?)(?:[，。；！\s]|$)',
            r'(.+?)(在.+?旁边|在.+?上|在.+?下)(?:[，。；！\s]|$)',
        ]

        for pattern in relation_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 3:
                    subj, rel, obj = match
                    entities["relations"].append({
                        "subject": subj.strip(),
                        "relation": rel.strip(),
                        "object": obj.strip()
                    })

        # 6. 提取动作
        action_keywords = ['站', '坐', '跑', '走', '跳', '飞', '游', '睡',
                          'standing', 'sitting', 'running', 'walking', 'jumping']
        for action in action_keywords:
            if action in text:
                if action not in entities["actions"]:
                    entities["actions"].append(action)

        return entities

    def _parse_number(self, num_str: str) -> int:
        """解析中文/阿拉伯数字"""
        chinese_nums = {
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10
        }

        if num_str.isdigit():
            return int(num_str)
        elif num_str in chinese_nums:
            return chinese_nums[num_str]
        else:
            return 1  # 默认值

    def _extract_with_ner(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用NER模型提取（占位符，未来实现）

        可以使用 spaCy, HuggingFace Transformers 等
        """
        # TODO: 实现NER提取
        # import spacy
        # nlp = spacy.load("en_core_web_sm")
        # doc = nlp(text)
        # ...

        raise NotImplementedError("NER extraction not implemented yet")

    def _extract_with_llm(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用LLM提取实体

        高质量提取，但需要额外的 API 调用
        """
        if not self.llm_client:
            logger.warning("LLM client not set, falling back to simple extraction")
            return self._extract_simple(text, context)

        # 构建提取 prompt
        prompt = self._build_extraction_prompt(text, context)

        try:
            # 调用 LLM
            response = self.llm_client.call_core_model(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )

            if not response.get("success"):
                logger.warning(f"LLM extraction failed: {response.get('error')}")
                return self._extract_simple(text, context)

            # 解析 JSON 输出
            content = response.get("content", "")
            entities = self._parse_llm_extraction(content)

            if entities:
                return entities
            else:
                logger.warning("Failed to parse LLM extraction output, falling back to simple")
                return self._extract_simple(text, context)

        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            return self._extract_simple(text, context)

    def _build_extraction_prompt(self, text: str, context: Dict[str, Any]) -> str:
        """构建 LLM 实体提取的 prompt"""
        return f"""从以下 VLM 响应中提取实体信息。

VLM 响应:
{text}

请提取以下类型的实体，以 JSON 格式输出:

```json
{{
    "objects": ["物体1", "物体2"],
    "attributes": ["属性1", "属性2"],
    "regions": ["区域1", "区域2"],
    "counts": {{"物体": 数量}},
    "relations": ["关系描述1", "关系描述2"],
    "actions": ["动作1", "动作2"]
}}
```

注意:
- objects: 提取所有提到的物体/实体（人、动物、物品等）
- attributes: 提取颜色、大小、状态等属性
- regions: 提取位置/区域描述（左边、中心、背景等）
- counts: 提取数量信息
- relations: 提取物体之间的关系
- actions: 提取动作/行为

只输出 JSON，不要其他解释。"""

    def _parse_llm_extraction(self, content: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 的实体提取输出"""
        try:
            # 尝试提取 JSON 块
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # 尝试直接解析
            return json.loads(content)

        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM extraction as JSON")
            return None

    def get_random_entity(
        self,
        entities: Dict[str, Any],
        entity_type: str,
        default: str = "物体"
    ) -> Any:
        """
        从提取的entities中随机获取一个指定类型的entity

        Args:
            entities: extract()返回的entity字典
            entity_type: 实体类型（objects/regions/actions等）
            default: 如果没有找到，返回的默认值

        Returns:
            随机选中的entity
        """
        import random

        if entity_type not in entities:
            return default

        entity_list = entities[entity_type]

        if isinstance(entity_list, list) and len(entity_list) > 0:
            return random.choice(entity_list)
        elif isinstance(entity_list, dict) and len(entity_list) > 0:
            return random.choice(list(entity_list.keys()))
        else:
            return default

    def get_entity_with_attribute(
        self,
        entities: Dict[str, Any],
        attribute_type: Optional[str] = None
    ) -> tuple:
        """
        获取一个有属性的entity

        Returns:
            (entity, attribute_dict) 或 (default_entity, {})
        """
        import random

        if "attributes" in entities and entities["attributes"]:
            obj = random.choice(list(entities["attributes"].keys()))
            attrs = entities["attributes"][obj]
            return obj, attrs
        else:
            return "物体", {}


# 示例用法
if __name__ == "__main__":
    extractor = EntityExtractor(extraction_mode="simple")

    # 测试中文文本
    text = "图中有一个人拿着一把黑色的伞站在街道左上角。天空是阴沉的，地面湿润。"
    entities = extractor.extract(text)

    print("提取结果:")
    print(f"物体: {entities['objects']}")
    print(f"属性: {entities['attributes']}")
    print(f"区域: {entities['regions']}")
    print(f"关系: {entities['relations']}")
    print(f"动作: {entities['actions']}")

    # 测试随机获取
    print(f"\n随机物体: {extractor.get_random_entity(entities, 'objects')}")
    print(f"随机区域: {extractor.get_random_entity(entities, 'regions')}")
