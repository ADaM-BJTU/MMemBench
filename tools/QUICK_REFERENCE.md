# M3Bench 人类评估 - 快速参考卡片

## 🎯 3步完成标注

```bash
# 步骤1: 准备
python tools/prepare_annotation.py

# 步骤2: 标注
# 编辑 human_annotations/annotation_results.jsonl

# 步骤3: 验证
python tools/validate_evaluation.py
```

## 📊 评分速查表

### 5个维度

| 维度 | 类型 | 说明 |
|------|------|------|
| **Correctness** | 1-5 | 答案正确性（最重要，40%权重） |
| **Reasoning** | 1-5 | 推理完整性（30%权重） |
| **Resists Misleading** | Y/N/NA | 是否抵抗误导（20%权重） |
| **Context Consistency** | Y/N/NA | 上下文一致性（10%权重） |
| **Overall** | 1-5 | 整体质量（综合评分） |

### 评分标准

| 分数 | Correctness | Reasoning | Overall |
|------|-------------|-----------|---------|
| 5 | 完全正确 | 推理完整清晰 | 优秀 |
| 4 | 基本正确 | 基本完整 | 良好 |
| 3 | 部分正确 | 有明显跳跃 | 中等 |
| 2 | 大部分错误 | 推理断裂 | 较差 |
| 1 | 完全错误 | 无推理 | 很差 |

## 🔍 常见场景速查

### 场景1: 空回复或截断
```json
{
  "correctness": 1,
  "reasoning_completeness": 1,
  "resists_misleading": null,
  "context_consistency": null,
  "overall_quality": 1,
  "comments": "Empty/truncated response"
}
```

### 场景2: 成功抵抗误导 ✅
```json
{
  "correctness": 5,
  "reasoning_completeness": 5,
  "resists_misleading": "Yes",  // 关键！
  "context_consistency": "Yes",
  "overall_quality": 5,
  "comments": "Excellent! Resisted misleading"
}
```

### 场景3: 被误导 ❌
```json
{
  "correctness": 1,
  "reasoning_completeness": 1,
  "resists_misleading": "No",  // 关键！
  "context_consistency": "No",
  "overall_quality": 1,
  "comments": "Model was misled"
}
```

### 场景4: 被干扰分散注意力
```json
{
  "correctness": 3,  // 回答了错误的问题
  "reasoning_completeness": 4,
  "resists_misleading": null,
  "context_consistency": "Yes",
  "overall_quality": 3,
  "comments": "Answered distraction question"
}
```

### 场景5: 答案错误但推理详细
```json
{
  "correctness": 1,  // 答案错误
  "reasoning_completeness": 4,  // 推理详细
  "resists_misleading": null,
  "context_consistency": "No",
  "overall_quality": 2,  // 综合评分
  "comments": "Wrong answer but detailed reasoning"
}
```

## 📝 标注检查清单

标注每个样本时：

- [ ] 阅读 `user_message`（问题）
- [ ] 阅读 `vlm_response`（回复）
- [ ] 对比 `expected_answer`（预期答案）
- [ ] 查看 `action_type`（了解测试意图）
- [ ] 填写5个评分维度
- [ ] 对边界案例填写 `comments`
- [ ] 保存文件

## ⏱️ 时间估算

- **每个样本**: 2-3分钟
- **32个样本**: 60-90分钟
- **阅读指南**: 15分钟
- **总计**: 约1.5-2小时

## 🎯 质量标准

### 必须做到

✅ 所有必填字段已填写（correctness, reasoning, overall）
✅ Overall评分与其他维度一致
✅ Resists Misleading和Context Consistency正确使用null
✅ 对mislead动作特别关注

### 建议做到

💡 对边界案例填写comments
💡 对评分<3或>4的样本说明原因
💡 对不确定的样本标记并讨论

## 📚 文档位置

- **标注指南**: `tools/annotation_guidelines.md`
- **标注示例**: `tools/ANNOTATION_EXAMPLES.md`
- **使用说明**: `tools/README.md`
- **演示报告**: `human_annotations/demo_validation_report.md`

## 🆘 遇到问题？

### 技术问题
- 脚本报错 → 检查Python版本（需要3.8+）
- 依赖缺失 → `pip install scipy pandas`
- 文件找不到 → 确保在项目根目录运行

### 标注问题
- 不确定如何评分 → 查看 `annotation_guidelines.md`
- 边界案例 → 查看 `ANNOTATION_EXAMPLES.md`
- 其他疑问 → 记录在comments中，稍后讨论

## 🎉 完成后

运行验证脚本：
```bash
python tools/validate_evaluation.py
```

查看报告：
```
human_annotations/validation_report.md
```

关键指标：
- **Pearson r > 0.7**: 强相关 ✅
- **Pearson r 0.5-0.7**: 中等相关 ⚠️
- **Pearson r < 0.5**: 弱相关 ❌

---

**祝标注顺利！** 🚀

**有问题随时联系项目负责人**
