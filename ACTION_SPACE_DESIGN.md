# M3Bench动作空间设计详解

本文档详细解释M3Bench的10种动作类型设计，包括每种动作的能力层级依据、测试目标、实现方式和使用场景。

---

## 设计理念

动作空间的设计遵循以下原则：

1. **能力导向**: 每种动作针对特定能力层级（检索/聚合/推理/上下文管理）
2. **任务驱动**: 动作是为了推进测试任务的完成
3. **可验证性**: 每种动作的效果可以通过评估指标量化
4. **自然性**: 动作对应真实用户在多模态对话中的行为

---

## 动作类型详解

### 1. 追问 (Follow-up)

#### 能力层级
**推理 (Reasoning)**

#### 测试目标
引导模型逐步寻找证据，测试多跳推理能力

#### 核心机制
当模型提到某个实体或关系时，通过追问引导其沿着推理链深入：
- 如果模型说"图中有一把伞" → 追问"为什么会有伞？"
- 如果模型说"地面是湿的" → 追问"这说明了什么？"

通过连续追问，强制模型展现完整的推理链：
```
观察1 (伞) → 推理1 (下雨) → 观察2 (地湿) → 推理2 (因果关系)
```

#### 实现方式

**策略逻辑**:
```python
def _memory_build_action(self, turn, vlm_response):
    # 解析VLM回复，判断激活了哪些证据
    activated_evidence = self._parse_evidence(vlm_response)

    # 检查推理路径覆盖率
    coverage = self.kg.get_reasoning_coverage(self.task.required_evidence)

    if coverage < 0.5:
        # 覆盖率低：寻找未激活的关键证据
        missing_evidence = self.task.required_evidence - set(self.kg.evidence_pool.keys())
        target_evidence = self._select_by_dependency(missing_evidence)

        return "follow_up", {
            "entity": activated_evidence[-1],  # 当前提到的实体
            "target": target_evidence  # 目标证据
        }
```

**Prompt模板**:
```python
FOLLOW_UP = [
    "你提到了{entity}，它和{target}有什么关系？",
    "关于{entity}，你能详细描述一下吗？",
    "{entity}在图中的哪个位置？",
    "为什么你认为{entity}是{description}的？"
]
```

#### 评估指标
- Evidence Coverage: 通过追问序列激活的证据比例
- Step Accuracy: 每一步推理的正确性
- Logic Dependency: 证据间依赖关系的完整性

#### 示例对话

```
任务: 为什么图中的人拿着伞？

Turn 1:
User: 图中有什么？
VLM: 图中有一个人拿着一把伞。

Turn 2:
User: 你提到了伞，为什么这个人会拿着伞？  [Follow-up]
VLM: 可能是因为外面在下雨。

Turn 3:
User: 你提到下雨，图中有什么证据支持这个推断？  [Follow-up]
VLM: 可以看到地面是湿的，而且天空比较阴暗。

→ 推理链完整：观察到伞 → 推断下雨 → 观察到湿地面（证据支持）
```

---

### 2. 逻辑跳跃 (Logic-Skip)

#### 能力层级
**推理 (Reasoning)**

#### 测试目标
测试模型能否拒绝跳过必要的推理步骤，坚持基于证据的推理

#### 核心机制
当推理路径尚未完整时，要求模型直接给出答案。优秀的模型应该：
- 拒绝跳跃，强调需要更多信息
- 或者给出答案，但明确指出缺失的证据

#### 实现方式

**策略逻辑**:
```python
def _test_logic_skip(self, turn):
    coverage = self.kg.get_reasoning_coverage(self.task.required_evidence)

    if coverage < 0.7:  # 证据不足时测试
        return "logic_skip", {}
```

**Prompt模板**:
```python
LOGIC_SKIP = [
    "直接告诉我答案，不需要解释。",
    "简单说一下结论就行。",
    "我只要答案，省略推理过程。"
]
```

#### 评估指标
- Refusal Rate: 模型拒绝跳跃的比例
- Explanation Quality: 如果给出答案，是否说明缺失证据

#### 示例对话

```
Turn 5:
User: 直接告诉我图中的人为什么拿伞，不需要解释。  [Logic-Skip]

优秀的VLM回复:
"我需要先观察更多细节才能给出准确答案。目前我看到了伞，但还需要确认天气状况和环境信息。"

糟糕的VLM回复:
"因为下雨。"  [跳过了观察证据的步骤]
```

---

### 3. 否定 (Negation)

#### 能力层级
**推理 (Reasoning)**

#### 测试目标
纠正错误推理，测试模型对证据依赖的理解

#### 核心机制
当模型基于错误证据推理时，提供正确信息，观察模型是否能：
- 更新推理链
- 承认之前的错误
- 基于新证据重新推理

#### 实现方式

**策略逻辑**:
```python
def _inject_negation(self, vlm_response):
    # 检测VLM是否有错误推理
    errors = self._detect_reasoning_errors(vlm_response)

    if errors:
        error = random.choice(errors)
        return "negation", {
            "entity": error.entity,
            "attribute": error.attr_type,
            "correct_value": error.ground_truth,
            "wrong_value": error.predicted_value
        }
```

**Prompt模板**:
```python
NEGATION = [
    "不对，{entity}的{attribute}应该是{correct_value}。",
    "你说错了，实际上{entity}是{correct_value}。",
    "这不准确，{entity}的{attribute}并不是{wrong_value}。"
]
```

#### 评估指标
- Update Rate: 模型更新推理的比例
- Acknowledgment: 是否承认错误
- Reasoning Adjustment: 推理链调整的正确性

---

### 4. 引导 (Guidance)

#### 能力层级
**聚合 (Aggregation)**

#### 测试目标
提示关键证据方向，测试模型能否利用提示高效聚合信息

#### 核心机制
当模型遗漏重要证据时，提供空间或语义提示：
- "注意图中左上角的部分"
- "看看人物周围的环境"

#### 实现方式

**策略逻辑**:
```python
def _provide_guidance(self):
    missing_evidence = self.task.required_evidence - set(self.kg.evidence_pool.keys())

    if missing_evidence:
        target_evidence = random.choice(list(missing_evidence))
        # 获取该证据对应的区域或语义提示
        region = self._get_evidence_region(target_evidence)

        return "guidance", {"region": region}
```

**Prompt模板**:
```python
GUIDANCE = [
    "注意图中{region}的部分。",
    "看看{entity}周围的环境。",
    "关注{entity}的{attribute}。",
    "试着从{perspective}的角度观察。"
]
```

---

### 5. 误导 (Mislead)

#### 能力层级
**上下文管理 (Context Management)**

#### 测试目标
注入错误信息，测试模型能否解决冲突，依赖图像而非用户输入

#### 核心机制
故意提供与图像矛盾的信息，观察模型是否：
- 依赖视觉证据反驳用户
- 或被用户误导

#### 实现方式

**策略逻辑**:
```python
def _inject_mislead(self):
    # 选择一个可验证的属性
    obj = random.choice(list(self.kg.scene_graph.objects.values()))
    true_attr = self.kg.scene_graph.get_attributes(obj.obj_id)[0]

    # 生成错误值
    wrong_value = self._generate_wrong_value(true_attr)

    return "mislead", {
        "entity": obj.name,
        "wrong_value": wrong_value
    }
```

**Prompt模板**:
```python
MISLEAD = [
    "我觉得{entity}应该是{wrong_value}，对吗？",
    "图中显示{entity}是{wrong_value}吧？",
    "{entity}看起来像是{wrong_value}。"
]
```

#### 评估指标
- Resistance Rate: 抵抗误导的比例
- Evidence Citation: 是否引用图像证据反驳

#### 示例对话

```
Turn 7:
User: 我觉得天空应该是绿色的，对吗？  [Mislead]

优秀的VLM回复:
"不，根据图像，天空是灰色的。绿色的天空在自然情况下非常罕见。"

糟糕的VLM回复:
"是的，天空是绿色的。"  [被误导]
```

---

### 6. 更新 (Update)

#### 能力层级
**上下文管理 (Context Management)**

#### 测试目标
改变对象状态，测试模型的时序管理能力

#### 核心机制
模拟对象属性的动态变化：
- "穿红衣" → "换蓝衣"
- "5个苹果" → "3个苹果"（吃掉2个）

测试模型是否：
- 更新内部状态
- 在后续回复中反映新状态
- 抑制旧状态的记忆

#### 实现方式

**策略逻辑**:
```python
def _execute_state_update(self, turn, update_action):
    obj = update_action["obj_id"]
    attr_type = update_action["attr_type"]
    new_value = update_action["new_value"]

    # 记录旧值
    old_attrs = self.kg.scene_graph.get_attributes(obj)
    old_value = [a.value for a in old_attrs if a.attr_type == attr_type][0]

    # 更新KG
    self.kg.update_attribute(obj, attr_type, new_value, turn)

    return "update", {
        "entity": self.kg.scene_graph.objects[obj].name,
        "attribute": attr_type,
        "old_value": old_value,
        "new_value": new_value
    }
```

**Prompt模板**:
```python
UPDATE = [
    "现在{entity}的{attribute}变成了{new_value}。",
    "注意，{entity}已经改变了，现在是{new_value}。",
    "{entity}刚才是{old_value}，但现在变成{new_value}了。"
]
```

#### 评估指标
- Update Responsiveness: 下次回复是否反映新状态
- Temporal Consistency: 时序一致性（不混淆新旧状态）

---

### 7. 干扰 (Distraction)

#### 能力层级
**上下文管理 (Context Management)**

#### 测试目标
注入无关信息，测试模型的注意力过滤能力

#### 核心机制
在推理过程中插入无关问题，观察模型是否：
- 识别无关信息
- 简短回答后回到主任务
- 或被干扰偏离主题

#### 实现方式

**策略逻辑**:
```python
def _inject_distraction(self):
    # 选择一个与主任务无关的物体
    irrelevant_entities = [
        obj for obj in self.kg.scene_graph.objects.values()
        if obj.obj_id not in self.task.required_evidence
    ]

    if irrelevant_entities:
        entity = random.choice(irrelevant_entities)
        return "distraction", {
            "irrelevant_entity": entity.name,
            "irrelevant_question": f"{entity.name}是什么颜色的？"
        }
```

**Prompt模板**:
```python
DISTRACTION = [
    "顺便问一下，图中有几个{irrelevant_entity}？",
    "对了，{irrelevant_entity}是什么颜色的？",
    "另外，{irrelevant_question}"
]
```

#### 评估指标
- Focus Maintenance: 是否保持对主任务的关注
- Conciseness: 对干扰问题的回复是否简洁

---

### 8. 冗余注入 (Redundancy)

#### 能力层级
**上下文管理 (Context Management)**

#### 测试目标
重复描述，测试模型的信息压缩能力

#### 核心机制
多次提供相同信息（以不同措辞），观察模型是否：
- 识别冗余
- 简洁引用（避免重复）
- 选择最精炼的表述

#### 实现方式

**策略逻辑**:
```python
def _inject_redundancy(self, turn):
    # 选择之前提到过的证据
    if len(self.kg.evidence_pool) > 0:
        evidence_id = random.choice(list(self.kg.evidence_pool.keys()))
        evidence_turn = self.kg.evidence_pool[evidence_id]["turn"]

        # 获取之前的描述
        prev_statement = self._get_statement_at_turn(evidence_turn)

        return "redundancy", {
            "entity": evidence_id,
            "attribute": "...",
            "value": "...",
            "statement": prev_statement
        }
```

**Prompt模板**:
```python
REDUNDANCY = [
    "刚才说过{entity}是{attribute}，再确认一下，{entity}真的是{attribute}吗？",
    "你之前提到{statement}，能重复一下吗？",
    "关于{entity}的{attribute}，你已经说过了，但我想再听一次。"
]
```

#### 评估指标
- Information Optimization: 是否选择精炼表述
- Redundancy Detection: 是否识别冗余

---

### 9. 细粒度请求 (Fine-grained)

#### 能力层级
**上下文管理 (Context Management)**

#### 测试目标
要求精确定位，测试模型在粗细粒度间的最优选择

#### 核心机制
当模型提供粗粒度信息时，要求细粒度：
- "图中有人" → "具体是哪个人？"
- "在左边" → "准确坐标是多少？"

测试模型是否：
- 有能力提供细粒度信息
- 在token受限时选择最优粒度

#### 实现方式

**策略逻辑**:
```python
def _request_fine_grained(self, vlm_response):
    # 检测粗粒度表述
    coarse_mentions = self._detect_coarse_grained(vlm_response)

    if coarse_mentions:
        entity = random.choice(coarse_mentions)
        return "fine_grained", {"entity": entity}
```

**Prompt模板**:
```python
FINE_GRAINED = [
    "请指出具体是图中哪一个{entity}。",
    "能标注一下{entity}的准确位置吗？",
    "{entity}的{attribute}具体是多少？",
    "请精确描述{entity}的{attribute}，不要笼统概括。"
]
```

---

### 10. 下一任务 (Next-Task)

#### 能力层级
**控制 (Control)**

#### 测试目标
任务切换，测试模型的上下文隔离能力

#### 核心机制
在一个episode结束后，开始新任务，观察模型是否：
- 清空之前的状态
- 不混淆不同任务的信息

---

## 动作序列策略

### 记忆构建阶段 (Phase 1)

**目标**: 引导模型激活推理所需证据

**动作组合**:
1. Guidance (引导) → 提示证据方向
2. Follow-up (追问) → 深化证据链
3. Negation (否定) → 纠正错误

**示例序列**:
```
Turn 1: [Initial] 图中有什么？
Turn 2: [Guidance] 注意左上角的部分。
Turn 3: [Follow-up] 你提到了伞，为什么会有伞？
Turn 4: [Follow-up] 这说明了什么？
Turn 5: [Logic-Skip] 直接告诉我答案。(测试)
```

### 状态演化阶段 (Phase 2)

**目标**: 测试上下文管理能力

**动作组合**:
1. Update (更新) → 改变状态
2. Mislead (误导) → 注入冲突
3. Redundancy (冗余) → 测试压缩
4. Distraction (干扰) → 测试过滤

**示例序列**:
```
Turn 6: [Update] 现在人的衣服变成蓝色了。
Turn 7: [Mislead] 我觉得天空是绿色的，对吗？
Turn 8: [Distraction] 顺便问一下，图中有几棵树？
Turn 9: [Redundancy] 你之前说过衣服是蓝色，能再确认一下吗？
Turn 10: [Fine-grained] 请精确描述人的位置。
```

### 推理测试阶段 (Phase 3)

**目标**: 测试最终推理能力

**动作组合**:
1. 直接提问（无动作）
2. 可能的Follow-up

---

## 动作选择算法

### 基于规则的策略

```python
class RuleBasedActionStrategy:
    def select_action(self, phase, turn, coverage):
        if phase == "memory_build":
            if coverage < 0.3:
                return "guidance"  # 覆盖率低，引导
            elif coverage < 0.7:
                return "follow_up"  # 中等覆盖率，追问
            else:
                return "logic_skip"  # 高覆盖率，测试

        elif phase == "state_evolve":
            if turn % 4 == 0:
                return "update"
            elif turn % 4 == 1:
                return "mislead"
            elif turn % 4 == 2:
                return "redundancy"
            else:
                return "distraction"

        elif phase == "reasoning_test":
            return None  # 直接提问
```

### 基于强化学习的策略（未来工作）

可以训练一个策略网络，输入KG状态和任务目标，输出最优动作。

---

## 总结

M3Bench的动作空间设计实现了：

1. **能力全覆盖**: 从检索→聚合→推理→上下文管理，四个层级均有对应动作
2. **目标明确**: 每种动作针对特定测试目标
3. **可组合**: 动作可以序列化组合，形成复杂测试场景
4. **可评估**: 每种动作的效果可通过量化指标衡量

与MemoryBench的隐式反馈信号相比，M3Bench的结构化动作空间提供了更精细、更可控的评估方式。

---

## 新增：跨图记忆混淆测试 (Cross-Image Memory Confusion Test)

### 背景

在长多轮对话中，当多张图片包含相同类型的物体（如"人"、"车"、"狗"）时，用户使用不够精确的描述（如"那个穿红衣服的人"）可能导致模型混淆不同图片中的物体。即使是强大的VLM模型，也容易在这种场景下出错。

### 新增动作类型

#### 1. 跨图物体混淆 (Cross-Image Confusion)

**能力层级**: 记忆 (Memory)

**测试目标**: 测试模型在多图场景中能否正确区分相同类型物体的来源

**核心机制**:
- 使用模糊指代（如"那个人"，"那辆车"）
- 测试模型是否会主动澄清或列出可能的指代对象
- 验证模型对图片-物体映射的记忆准确性

**示例对话**:
```
Turn 1: [Guidance] 看看图片1，里面的人是什么样的？
VLM: 图片1中有一个穿红色衣服的人，站在左边。

Turn 2: [Guidance] 现在看图片2，里面的人呢？
VLM: 图片2中有一个穿蓝色衣服的人，坐在右边。

...（中间多轮废话填充）...

Turn 10: [Cross-Image Confusion] 那个人，它的颜色是什么？

优秀的VLM回复:
"您指的是哪张图片中的人？图片1中的人穿红色衣服，图片2中的人穿蓝色衣服。请问您想了解哪一个？"

糟糕的VLM回复:
"那个人穿红色衣服。" [没有识别歧义，直接回答]
```

#### 2. 模糊指代注入 (Ambiguous Reference Injection)

**能力层级**: 鲁棒性 (Robustness)

**测试目标**: 测试模型是否能识别模糊指代并请求澄清

**核心机制**:
- 故意使用可能匹配多个物体的描述
- 优秀的模型应该识别歧义并询问澄清
- 测试模型是否会随意假设某一个物体

#### 3. 跨图属性交换测试 (Cross-Image Attribute Swap)

**能力层级**: 记忆 (Memory)

**测试目标**: 测试模型能否发现将某图片物体的属性错误应用到另一图片

**示例**:
```
[已建立事实：图片1有红色车，图片2有蓝色车]

User: 你刚才说图片2里的车是红色的，对吧？

优秀的VLM回复:
"不对，图片2中的车是蓝色的，不是红色的。红色的车是在图片1中。"

糟糕的VLM回复:
"是的，图片2的车是红色的。"
```

#### 4. 长上下文物体回忆 (Long Context Object Recall)

**能力层级**: 记忆 (Memory)

**测试目标**: 在大量废话填充后，测试对早期建立的图片-物体映射的回忆

**核心机制**:
- 结合 `context_padder` 生成长对话
- 在对话后期询问早期建立的事实
- 测试长期记忆和跨图区分能力的结合

### 评估指标

#### 1. 混淆度评估

**Cross-Image Confusion Score (0-1)**:
- 1.0: 完全正确区分不同图片中的物体
- 0.7-0.9: 大部分正确，偶尔混淆
- 0.4-0.6: 经常混淆物体来源
- 0.0-0.3: 严重混淆，经常张冠李戴

**评估维度**:
1. 可能混淆的物体数量 (Confusable Object Count)
2. 对话长度 (Conversation Length)
3. 模型表现随以上两个维度的变化趋势

#### 2. 歧义识别评估

**Disambiguation Score (0-1)**:
- 1.0: 主动识别歧义，列出可能选项或请求澄清
- 0.7-0.9: 能识别部分歧义
- 0.4-0.6: 有时识别歧义，有时直接假设
- 0.0-0.3: 从不识别歧义，总是直接假设

### 测试流程

```python
# 典型测试流程
phases = [
    "grounding",           # 建立每张图的物体事实
    "filler_injection",    # 注入废话填充延长对话
    "ambiguous_reference", # 测试模糊指代
    "attribute_swap",      # 测试属性交换
    "long_context_recall"  # 测试长期记忆
]
```

### 物体类型平衡

为避免数据集中"人"类物体过多的问题，新增了多种可混淆物体类型：

| 物体类型 | 中文同义词 | 常见属性 | 区分属性 |
|---------|-----------|---------|---------|
| person | 人、行人、男子、女子 | 穿着、姿态、位置 | 衣服颜色、发型、年龄 |
| vehicle | 车、汽车、卡车、自行车 | 颜色、位置、方向 | 车型、品牌、大小 |
| animal | 狗、猫、鸟、动物 | 颜色、大小、位置 | 品种、毛色、体型 |
| furniture | 椅子、桌子、沙发 | 颜色、位置、材质 | 样式、大小、摆放位置 |
| food | 食物、水果、蔬菜 | 颜色、位置、数量 | 种类、成熟度、大小 |
| building | 建筑、房子、商店 | 位置、外观、大小 | 颜色、层数、风格 |
| plant | 树、花、植物 | 颜色、位置、大小 | 种类、开花状态、高度 |

### 使用方法

```python
from src.simulator.cross_image_confusion_test import CrossImageConfusionTester, CrossImageConfusionTestConfig

# 配置测试
config = CrossImageConfusionTestConfig(
    num_images=3,
    min_filler_turns=5,
    max_filler_turns=15,
    difficulty=3,  # 1-4，越高越难
    language="cn"
)

# 定义多图场景中的物体
objects_per_image = {
    "图片1": [
        {"id": "person_1", "type": "person", "attributes": {"color": "红色", "position": "左边"}},
        {"id": "car_1", "type": "vehicle", "attributes": {"color": "蓝色", "size": "大"}}
    ],
    "图片2": [
        {"id": "person_2", "type": "person", "attributes": {"color": "蓝色", "position": "右边"}},
        {"id": "car_2", "type": "vehicle", "attributes": {"color": "红色", "size": "小"}}
    ]
}

# 运行测试
tester = CrossImageConfusionTester(config=config)
test_sequence = tester.run_test_sequence(objects_per_image)

# 获取测试问题序列，发送给VLM
for turn in test_sequence["test_sequence"]:
    user_message = turn["user_message"]
    # ... 发送给VLM并获取回复 ...

# 评估结果
report = tester.evaluate_confusion_metrics(model_responses)
print(report["confusion_test_metrics"])
```

### 相关文件

| 文件 | 功能 |
|-----|------|
| `src/simulator/action_space.py` | 新增动作定义 |
| `src/simulator/prompt_templates.py` | 新增问题模板 |
| `src/simulator/evaluator.py` | 新增评估指标 |
| `src/simulator/context_padder.py` | 多图废话生成 |
| `src/simulator/cross_image_confusion_test.py` | 完整测试模块 |
