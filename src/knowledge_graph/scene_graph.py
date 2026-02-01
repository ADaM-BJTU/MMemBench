"""
Scene Graph数据结构
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Object:
    """物体"""
    obj_id: str
    name: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    image_region: Optional[Any] = None


@dataclass
class Attribute:
    """属性"""
    attr_type: str  # color, size, state, action, etc.
    value: Any
    confidence: float = 1.0


@dataclass
class Relation:
    """关系"""
    subj_id: str
    predicate: str
    obj_id: str
    confidence: float = 1.0


class SceneGraph:
    """场景图"""

    def __init__(self, image_id: str):
        self.image_id = image_id
        self.objects: Dict[str, Object] = {}
        self.attributes: Dict[str, List[Attribute]] = {}
        self.relations: List[Relation] = []

    def add_object(
        self,
        obj_id: str,
        name: str,
        bbox: Tuple[int, int, int, int],
        image_region: Optional[Any] = None
    ):
        """添加物体"""
        self.objects[obj_id] = Object(
            obj_id=obj_id,
            name=name,
            bbox=bbox,
            image_region=image_region
        )
        self.attributes[obj_id] = []

    def add_attribute(
        self,
        obj_id: str,
        attr_type: str,
        value: Any,
        confidence: float = 1.0
    ):
        """添加属性"""
        if obj_id not in self.attributes:
            self.attributes[obj_id] = []

        self.attributes[obj_id].append(
            Attribute(
                attr_type=attr_type,
                value=value,
                confidence=confidence
            )
        )

    def add_relation(
        self,
        subj_id: str,
        predicate: str,
        obj_id: str,
        confidence: float = 1.0
    ):
        """添加关系"""
        self.relations.append(
            Relation(
                subj_id=subj_id,
                predicate=predicate,
                obj_id=obj_id,
                confidence=confidence
            )
        )

    def get_object(self, obj_id: str) -> Optional[Object]:
        """获取物体"""
        return self.objects.get(obj_id)

    def get_attributes(self, obj_id: str) -> List[Attribute]:
        """获取物体的所有属性"""
        return self.attributes.get(obj_id, [])

    def get_relations(
        self,
        obj_id: Optional[str] = None,
        as_subject: bool = True
    ) -> List[Relation]:
        """
        获取关系

        Args:
            obj_id: 物体ID（None表示返回全部）
            as_subject: True表示作为主语，False表示作为宾语

        Returns:
            关系列表
        """
        if obj_id is None:
            return self.relations

        if as_subject:
            return [r for r in self.relations if r.subj_id == obj_id]
        else:
            return [r for r in self.relations if r.obj_id == obj_id]

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "image_id": self.image_id,
            "objects": {
                obj_id: {
                    "name": obj.name,
                    "bbox": obj.bbox
                }
                for obj_id, obj in self.objects.items()
            },
            "attributes": {
                obj_id: [
                    {"type": attr.attr_type, "value": attr.value}
                    for attr in attrs
                ]
                for obj_id, attrs in self.attributes.items()
            },
            "relations": [
                {
                    "subject": rel.subj_id,
                    "predicate": rel.predicate,
                    "object": rel.obj_id
                }
                for rel in self.relations
            ]
        }
