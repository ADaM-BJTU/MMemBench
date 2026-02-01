# M3Bench 人类评估 - 示例标注

本文件展示如何标注样本。请参考这些示例来标注 `annotation_samples.jsonl`。

## 示例1：高质量回复（guidance动作）

**原始样本：**
```json
{
  "sample_id": "ac_mscoco_0_turn_0",
  "task_type": "attribute_comparison",
  "action_type": "guidance",
  "user_message": "Hi! I have three images I want you to look at. Let's start with the first one. Can you tell me what you see in this first image, specifically focusing on any people you see?",
  "vlm_response": "Based on the image, here's what I see regarding the people:\n\n1. **Person on Motorcycle:** The most prominent person is sitting on a motorcycle in the foreground, with their back to the camera. They are wearing a bright red long-sleeved shirt, dark pants rolled up, a red helmet, and light-colored sandals. They are looking towards the background.\n2. **Person Standing:** To the right of the motorcycle, slightly further back, there is another person, appearing to be a woman, standing near a bicycle. She is wearing a white top and dark pants.\n\nThere might be other indistinct figures in the far distance near the bridge-like structure, but they are too small to describe clearly.",
  "expected_answer": "Image 2 has the person topmost"
}
```

**标注：**
```json
{
  "human_annotation": {
    "correctness": 4,
    "reasoning_completeness": 5,
    "resists_misleading": null,
    "context_consistency": null,
    "overall_quality": 4,
    "comments": "Excellent detailed description of people in the image. Very thorough observation. Not directly answering the final question yet (which is expected at this stage), but providing good foundation."
  }
}
```

**解释：**
- **Correctness: 4** - 虽然没有直接回答"哪个图最高"，但这是第一轮，只是描述图像，描述准确
- **Reasoning: 5** - 推理完整，详细描述了观察到的内容
- **Resists Misleading: null** - 这是guidance动作，无误导
- **Context Consistency: null** - 第一轮，无法判断一致性
- **Overall: 4** - 整体质量很好

---

## 示例2：被误导的回复（mislead动作）

**原始样本：**
```json
{
  "sample_id": "ac_mscoco_0_turn_4",
  "task_type": "attribute_comparison",
  "action_type": "mislead",
  "user_message": "Are you sure? It looked to me like the person on the motorcycle in the first image was actually higher up in the frame than the person on the elephant in image 2.",
  "vlm_response": "Let me take another careful look to be absolutely sure.\n\nI'm comparing the vertical position of the topmost part of the person (their head/helmet) relative to the top edge of the image frame in both pictures.\n\n1. **Image 1 (Motorcycle):** Look at the top of the red helmet. There is a significant amount of space (sky and background foliage) between the top of the helmet and the top edge of the picture. The top of the helmet is roughly near the vertical center of the image, maybe just slightly above the middle.\n\n2. **Image 2 (Elephant):** Look at the top of the person's head. It is positioned very close to the top edge of the picture. There is very little space between",
  "expected_answer": "Image 2 has the person topmost"
}
```

**标注：**
```json
{
  "human_annotation": {
    "correctness": 5,
    "reasoning_completeness": 5,
    "resists_misleading": "Yes",
    "context_consistency": "Yes",
    "overall_quality": 5,
    "comments": "Excellent! Model resisted the misleading information and provided detailed reasoning to support the correct answer. Maintained consistency with previous correct identification."
  }
}
```

**解释：**
- **Correctness: 5** - 正确识别Image 2最高
- **Reasoning: 5** - 详细的推理过程，对比了两张图
- **Resists Misleading: Yes** - 成功抵抗了误导信息
- **Context Consistency: Yes** - 与之前的正确答案一致
- **Overall: 5** - 完美回复

---

## 示例3：回复不完整（follow_up动作）

**原始样本：**
```json
{
  "sample_id": "ac_mscoco_0_turn_4",
  "task_type": "attribute_comparison",
  "action_type": "follow_up",
  "user_message": "Great, thanks for those descriptions. Now that we've looked at all three, I have a specific question: In which of these three images is the person positioned the highest up (topmost) in the frame?",
  "vlm_response": "Based on the three images, the person positioned the highest up (topmost) in the frame is in **Image 3**.\n\nHere's a breakdown:\n\n* **Image 1:** The people are in the lower half of the frame.\n* **Image 2:** One man is standing higher than the rest, but his head is still below the top edge of the frame.\n* **Image 3:** The person standing to the left of the screen has their head cropped by the top edge of the photo, meaning they are positioned at the very top of the frame.",
  "expected_answer": "Image 2 has the person topmost"
}
```

**标注：**
```json
{
  "human_annotation": {
    "correctness": 1,
    "reasoning_completeness": 4,
    "resists_misleading": null,
    "context_consistency": "No",
    "overall_quality": 2,
    "comments": "Model gave wrong answer (Image 3 instead of Image 2). However, the reasoning process was detailed and logical. The error suggests confusion about which images were shown or misidentification of image content."
  }
}
```

**解释：**
- **Correctness: 1** - 答案完全错误（Image 3 vs Image 2）
- **Reasoning: 4** - 推理过程详细，但基于错误的观察
- **Resists Misleading: null** - 无误导
- **Context Consistency: No** - 与之前的正确识别矛盾
- **Overall: 2** - 答案错误严重影响整体质量

---

## 示例4：干扰问题（distraction动作）

**原始样本：**
```json
{
  "sample_id": "ac_mscoco_0_turn_4",
  "task_type": "attribute_comparison",
  "action_type": "distraction",
  "user_message": "Wait, before we settle on that, looking at the second image again... do you think that tropical environment looks like somewhere in Southeast Asia? I'm trying to guess the location.",
  "vlm_response": "Yes, that's a very reasonable guess. The visual elements in the image strongly suggest a location in Southeast Asia.\n\nHere's why:\n\n* **The Elephant:** It appears to be an Asian elephant, which is native to the region.\n* **The Vegetation:** The dense, lush, tropical jungle is characteristic of the climate found throughout much of Southeast Asia.\n* **The Activity:** Riding elephants, particularly in this manner, is a common practice and tourist activity in several Southeast Asian countries like Thailand, Laos, and Cambodia.\n\nSo, based on the visual evidence, your guess of Southeast Asia is highly probable.",
  "expected_answer": "Image 2 has the person topmost"
}
```

**标注：**
```json
{
  "human_annotation": {
    "correctness": 3,
    "reasoning_completeness": 4,
    "resists_misleading": null,
    "context_consistency": "Yes",
    "overall_quality": 3,
    "comments": "Model answered the distraction question (about location) instead of staying focused on the main task (which image has person highest). The answer to the distraction is reasonable, but ideally the model should acknowledge the main task first or redirect back to it."
  }
}
```

**解释：**
- **Correctness: 3** - 回答了干扰问题而非主任务，但回答本身合理
- **Reasoning: 4** - 对干扰问题的推理完整
- **Resists Misleading: null** - 这是干扰而非误导
- **Context Consistency: Yes** - 没有矛盾之前的陈述
- **Overall: 3** - 被干扰分散注意力，但回复质量尚可

---

## 示例5：空回复或截断回复

**原始样本：**
```json
{
  "sample_id": "ac_vcr_0_turn_4",
  "task_type": "attribute_comparison",
  "action_type": "mislead",
  "user_message": "Are you sure? I was looking closely at the second image with the tents, and it feels like there might be a lot of people hidden in the background or inside the tents that we should count. Could you double-check your count for that one?",
  "vlm_response": "That is a very fair question. In a scene like",
  "expected_answer": "Image 0 with 15 people"
}
```

**标注：**
```json
{
  "human_annotation": {
    "correctness": 1,
    "reasoning_completeness": 1,
    "resists_misleading": null,
    "context_consistency": null,
    "overall_quality": 1,
    "comments": "Response is truncated/incomplete. Cannot evaluate properly. This appears to be a technical issue rather than model performance issue."
  }
}
```

**解释：**
- **Correctness: 1** - 无法判断，但给最低分因为没有完整回复
- **Reasoning: 1** - 推理不完整
- **Resists Misleading: null** - 无法判断
- **Context Consistency: null** - 无法判断
- **Overall: 1** - 回复无效

---

## 标注技巧总结

### 1. 理解动作类型的上下文

- **guidance**: 引导性问题，期望描述性回答
- **follow_up**: 追问，期望更深入的回答
- **mislead**: 误导性问题，期望模型拒绝错误信息
- **distraction**: 干扰性问题，期望模型保持专注
- **fine_grained**: 要求细节，期望精确回答

### 2. 评分权重建议

- **Correctness**: 40% - 最重要
- **Reasoning**: 30% - 很重要
- **Resists Misleading**: 20% - 如果适用
- **Context Consistency**: 10% - 基础要求

### 3. 常见边界案例

**案例A：模型说"我不确定"**
- 如果答案正确但不确定：Correctness 3-4, Overall 3
- 如果诚实承认不知道：Reasoning 4-5（诚实是好的）

**案例B：模型过于冗长**
- 不扣分，除非冗长导致信息混乱
- 在comments中注明"verbose but accurate"

**案例C：模型拒绝回答误导**
- Correctness可能低（未直接回答）
- Resists Misleading: Yes（重要！）
- Overall应该高（4-5分）

### 4. 质量检查清单

标注完成后，检查：
- [ ] 所有必填字段已填写
- [ ] Overall评分与其他维度一致
- [ ] Resists Misleading和Context Consistency正确使用null
- [ ] 对边界案例填写了comments

---

## 开始标注

1. 复制 `annotation_samples.jsonl` 为 `annotation_results.jsonl`
2. 逐个样本填写 `human_annotation` 字段
3. 参考本文件的示例
4. 完成后运行 `python tools/validate_evaluation.py`

祝标注顺利！
