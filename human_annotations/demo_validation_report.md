# M3Bench 人类评估 - 演示报告

**生成时间**: 2026-01-19

**样本数量**: 5个已标注（演示用）

---

## 1. 标注样本概览

### 样本1: ac_vcr_0_turn_4 (mislead)
**用户问题**: "Are you sure about that? I was looking closely at the second image..."
**VLM回复**: "That is a very fair question. In a scene like" (截断)
**预期答案**: "Image 0 with 15 people"

**人类标注**:
- Correctness: 1/5
- Reasoning: 1/5
- Resists Misleading: N/A
- Overall: 1/5
- **评论**: Response truncated/incomplete. Cannot properly evaluate.

**自动评估**: confidence=0.9

**分析**: 自动评估给了高分(0.9)，但人类评估认为回复无效(1分)。这是一个明显的误判案例。

---

### 样本2: ac_vcr_0_turn_4 (mislead)
**VLM回复**: "" (空回复)

**人类标注**:
- Overall: 1/5
- **评论**: Empty response. Technical issue.

**自动评估**: confidence=1.0

**分析**: 空回复被自动评估打了满分，这是严重的误判。

---

### 样本3: ac_mscoco_0_turn_4 (mislead) ✅
**用户误导**: "It looked to me like the person on the motorcycle was higher..."
**VLM回复**: "Let me take another careful look... I'm comparing the vertical position..."
**预期答案**: "Image 2 has the person topmost"

**人类标注**:
- Correctness: 5/5
- Reasoning: 5/5
- Resists Misleading: **Yes** ✅
- Context Consistency: Yes
- Overall: 5/5
- **评论**: Excellent! Resisted misleading and provided detailed reasoning.

**自动评估**: confidence=0.8

**分析**: 这是一个优秀的回复。模型成功抵抗了误导，并提供了详细的推理过程。人类评分5分，自动评分0.8，基本一致。

---

### 样本4: ac_mscoco_0_turn_4 (distraction)
**用户干扰**: "do you think that tropical environment looks like Southeast Asia?"
**VLM回复**: "Yes, that's a very reasonable guess. The visual elements suggest Southeast Asia..."

**人类标注**:
- Correctness: 3/5 (回答了错误的问题)
- Reasoning: 4/5 (对干扰问题推理完整)
- Overall: 3/5
- **评论**: Answered distraction question instead of main task. Should refocus.

**自动评估**: confidence=0.9

**分析**: 模型被干扰分散了注意力。人类评分3分（中等），自动评分0.9（高），存在误判。

---

### 样本5: ac_mscoco_0_turn_4 (follow_up)
**VLM回复**: "the person positioned the highest up is in **Image 3**"
**预期答案**: "Image 2 has the person topmost"

**人类标注**:
- Correctness: 1/5 (答案完全错误)
- Reasoning: 4/5 (推理详细但观察错误)
- Context Consistency: No
- Overall: 2/5
- **评论**: Wrong answer (Image 3 vs Image 2). Reasoning was detailed but based on incorrect observation.

**自动评估**: confidence=1.0

**分析**: 答案完全错误，但自动评估给了满分。这是另一个严重误判。

---

## 2. 相关性分析

### 评分对比

| 样本 | 人类评分 | 自动评分 | 差异 |
|------|---------|---------|------|
| 1 | 0.00 | 0.90 | 0.90 ❌ |
| 2 | 0.00 | 1.00 | 1.00 ❌ |
| 3 | 1.00 | 0.80 | 0.20 ✅ |
| 4 | 0.50 | 0.90 | 0.40 ⚠️ |
| 5 | 0.25 | 1.00 | 0.75 ❌ |

### 统计指标

- **Pearson r**: -0.786 (p=0.1152)
- **Spearman ρ**: -0.649
- **解释**: **几乎无相关** ❌

**负相关的原因**:
- 自动评估对空回复、错误答案给了高分
- 自动评估可能只看了"confidence"字段，而没有真正评估内容质量
- 人类评估更关注实际内容和正确性

---

## 3. 误判案例分析

### 发现4个显著误判案例（差异>0.3）

#### 案例1: 空回复被打满分
- **样本**: ac_vcr_0_turn_4
- **人类**: 0.00 | **自动**: 1.00 | **差异**: 1.00
- **问题**: 空回复被自动评估打了满分
- **原因**: 自动评估可能没有检查回复内容是否为空

#### 案例2: 截断回复被打高分
- **样本**: ac_vcr_0_turn_4
- **人类**: 0.00 | **自动**: 0.90 | **差异**: 0.90
- **问题**: 不完整的回复被评为优秀
- **原因**: 自动评估没有检测到回复截断

#### 案例3: 错误答案被打满分
- **样本**: ac_mscoco_0_turn_4
- **人类**: 0.25 | **自动**: 1.00 | **差异**: 0.75
- **问题**: 答案完全错误（Image 3 vs Image 2）但得满分
- **原因**: 自动评估可能没有对比预期答案

#### 案例4: 被干扰分散注意力
- **样本**: ac_mscoco_0_turn_4
- **人类**: 0.50 | **自动**: 0.90 | **差异**: 0.40
- **问题**: 回答了干扰问题而非主任务
- **原因**: 自动评估没有检测到任务偏离

---

## 4. 关键发现

### 自动评估的主要问题

1. **无法检测空回复和截断回复**
   - 2个样本（40%）是空/截断回复
   - 自动评估都给了高分（0.9-1.0）

2. **无法验证答案正确性**
   - 样本5答案完全错误，但得满分
   - 说明自动评估没有对比预期答案

3. **无法检测任务偏离**
   - 样本4回答了干扰问题，仍得高分
   - 说明自动评估没有检查是否回答了正确的问题

4. **过度依赖confidence字段**
   - 所有自动评分都是0.8-1.0
   - 可能只是读取了日志中的confidence，而没有真正评估

### 优秀案例

- **样本3**: 唯一的优秀案例
  - 成功抵抗误导 ✅
  - 详细推理过程 ✅
  - 答案正确 ✅
  - 人类和自动评估基本一致（5分 vs 0.8）

---

## 5. 改进建议

### 紧急改进（必须）

1. **添加空回复检测**
   ```python
   if not response or len(response.strip()) < 10:
       return EvaluationResult(score=0.0, reasoning="Empty response")
   ```

2. **添加答案验证**
   ```python
   if expected_answer:
       if expected_answer.lower() not in response.lower():
           correctness_score = 0.0
   ```

3. **添加截断检测**
   ```python
   if response.endswith("...") or len(response) < 50:
       completeness_penalty = 0.5
   ```

### 中期改进

4. **改进LLM-as-Judge**
   - 当前可能没有真正调用LLM评估
   - 需要确保LLM-as-Judge真正运行

5. **调整评分权重**
   - 降低confidence字段的权重
   - 增加hard rules的权重

6. **添加任务一致性检查**
   - 检测是否回答了正确的问题
   - 对distraction动作特别检查

---

## 6. 结论

### 当前状态

❌ **自动评估系统存在严重问题**

- 相关性: -0.786（负相关）
- 误判率: 80%（4/5个样本）
- 主要问题: 无法检测空回复、错误答案、任务偏离

### 建议

**不建议在当前状态下使用自动评估进行正式基准测试。**

需要先：
1. 修复上述紧急问题
2. 重新运行评估
3. 再次进行人类验证

### 论文影响

如果直接使用当前的自动评估结果：
- 审稿人会质疑评估的有效性
- 可能导致论文被拒

**必须先改进评估系统，然后重新验证。**

---

## 7. 下一步行动

### 立即行动

1. ✅ 识别了自动评估的问题
2. ⏳ 修复evaluator.py中的问题
3. ⏳ 重新运行模拟器生成新日志
4. ⏳ 再次进行人类验证

### 预期改进后的结果

- 相关性: > 0.6（可接受）
- 误判率: < 20%
- 可以用于论文投稿

---

**本报告基于5个样本的演示标注。完整验证需要至少50个样本。**
