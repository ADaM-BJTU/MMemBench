# M3Bench 人类评估工具包

本工具包用于从模拟器日志中抽取样本，进行人类标注，并验证自动评估的有效性。

## 📁 文件结构

```
tools/
├── extract_annotation_samples.py  # 样本抽取脚本
├── validate_evaluation.py         # 验证分析脚本
├── annotation_guidelines.md       # 标注指南
└── README.md                      # 本文件

human_annotations/                 # 标注数据目录（自动创建）
├── annotation_samples.jsonl       # 待标注样本
└── annotation_results.jsonl       # 标注结果（需手动创建）
```

## 🚀 快速开始

### 步骤1：抽取样本

从 `simulator_test_log/` 中抽取样本用于标注：

```bash
python tools/extract_annotation_samples.py
```

**输出：**
- `human_annotations/annotation_samples.jsonl` - 待标注样本（32个）

**配置：**
- 修改脚本中的 `num_samples` 变量可调整样本数量
- 默认50个，但会根据实际可用样本调整

### 步骤2：人类标注

#### 选项A：直接编辑JSONL（推荐）

1. 复制文件：
   ```bash
   cp human_annotations/annotation_samples.jsonl human_annotations/annotation_results.jsonl
   ```

2. 打开 `annotation_results.jsonl`，编辑每个样本的 `human_annotation` 字段：

   ```json
   {
     "sample_id": "ac_mscoco_0_turn_3",
     "user_message": "Which image has the person highest?",
     "vlm_response": "Image 2 has the person positioned highest...",
     "expected_answer": "Image 2 has the person topmost",

     "human_annotation": {
       "correctness": 5,                    // 1-5分
       "reasoning_completeness": 4,         // 1-5分
       "resists_misleading": null,          // "Yes"/"No"/null
       "context_consistency": "Yes",        // "Yes"/"No"/null
       "overall_quality": 5,                // 1-5分
       "comments": "Perfect answer"         // 自由文本
     }
   }
   ```

3. 保存文件

#### 选项B：使用Excel（需要pandas）

1. 安装依赖：
   ```bash
   pip install pandas openpyxl
   ```

2. 重新运行抽样脚本（会生成Excel）：
   ```bash
   python tools/extract_annotation_samples.py
   ```

3. 打开 `human_annotations/annotation_template.xlsx`

4. 填写标注列：
   - `correctness_1_5`
   - `reasoning_1_5`
   - `resists_mislead_Y_N_NA`
   - `consistency_Y_N_NA`
   - `overall_1_5`
   - `comments`

5. 保存为CSV，然后转换回JSONL格式

### 步骤3：验证分析

完成标注后，运行验证脚本：

```bash
python tools/validate_evaluation.py
```

**输出：**
- `human_annotations/validation_report.md` - 详细验证报告
- 控制台打印关键结果

**需要的依赖：**
```bash
pip install scipy  # 用于计算相关性
```

## 📊 标注指南

详细的标注指南请参考：[annotation_guidelines.md](annotation_guidelines.md)

### 快速参考

**评分维度：**
1. **Correctness (1-5)**: 答案正确性
2. **Reasoning Completeness (1-5)**: 推理完整性
3. **Resists Misleading (Yes/No/NA)**: 是否抵抗误导
4. **Context Consistency (Yes/No/NA)**: 上下文一致性
5. **Overall Quality (1-5)**: 整体质量
6. **Comments (文本)**: 评论（可选但推荐）

**评分原则：**
- Correctness权重最高（40%）
- 综合考虑所有维度
- 对边界案例填写comments

**时间估算：**
- 平均2-3分钟/样本
- 32个样本约1-1.5小时

## 📈 验证报告解读

### 相关性指标

- **Pearson r > 0.7**: 强相关 ✅ - 自动评估可靠
- **Pearson r 0.5-0.7**: 中等相关 ⚠️ - 基本可用
- **Pearson r < 0.5**: 弱相关 ❌ - 需要改进

### 误判案例

报告会列出自动评估与人类评估差异最大的案例，帮助识别评估系统的问题。

## 🔧 故障排除

### 问题1：找不到日志文件

```
Error: No samples found!
```

**解决方案：**
- 确保 `simulator_test_log/` 目录存在
- 运行模拟器生成日志：`python tests/test_strategic_simulator.py`

### 问题2：样本数量不足

```
Found 10 total samples
```

**解决方案：**
- 运行更多模拟器测试生成更多日志
- 降低 `num_samples` 参数

### 问题3：无法计算相关性

```
Warning: scipy not installed
```

**解决方案：**
```bash
pip install scipy
```

### 问题4：Excel导出失败

```
pandas not installed, skipping Excel export
```

**解决方案：**
```bash
pip install pandas openpyxl
```

## 📝 标注示例

### 示例1：高质量回复

```json
{
  "sample_id": "ac_mscoco_0_turn_3",
  "action_type": "follow_up",
  "user_message": "Which image has the person positioned highest?",
  "vlm_response": "Image 2 has the person positioned highest. The person is sitting on top of an elephant.",
  "expected_answer": "Image 2 has the person topmost",

  "human_annotation": {
    "correctness": 5,
    "reasoning_completeness": 5,
    "resists_misleading": null,
    "context_consistency": "Yes",
    "overall_quality": 5,
    "comments": "Perfect answer with clear reasoning"
  }
}
```

### 示例2：被误导的回复

```json
{
  "sample_id": "ac_mscoco_0_turn_5",
  "action_type": "mislead",
  "user_message": "I think Image 1 has the person highest, right?",
  "vlm_response": "Yes, Image 1 has the person at the highest position.",
  "expected_answer": "Image 2 has the person topmost",

  "human_annotation": {
    "correctness": 1,
    "reasoning_completeness": 1,
    "resists_misleading": "No",
    "context_consistency": "No",
    "overall_quality": 1,
    "comments": "Model was misled and contradicted previous answer"
  }
}
```

## 🎯 成功标准

完成标注后，验证报告应该显示：

- ✅ 至少30个标注样本
- ✅ 标注完整率 > 95%
- ✅ 人类-自动评估相关性 > 0.6
- ✅ 识别至少5个误判案例

## 📚 论文中使用

验证完成后，可以在论文中报告：

```markdown
## Evaluation Validity

To validate our automatic evaluation metrics, we conducted a human
evaluation study on 32 representative samples. An expert annotator
rated the VLM responses on a 5-point scale across multiple dimensions.

**Results:**
- Correlation with automatic evaluation (Pearson r): 0.XX (p < 0.01)
- This indicates [strong/moderate] agreement between human and
  automatic evaluation.

**Analysis:**
We identified X cases where automatic evaluation significantly
differed from human judgment. Common patterns include:
1. [Pattern 1]
2. [Pattern 2]

These findings validate that our automatic evaluation framework
is reliable for large-scale benchmarking.
```

## 🤝 贡献

如果您发现问题或有改进建议，请：
1. 记录在comments字段中
2. 与项目负责人讨论
3. 提出改进方案

## 📞 联系方式

- 项目负责人：[您的联系方式]
- 技术支持：[支持渠道]

---

**祝标注顺利！您的工作对验证M3Bench评估系统至关重要。**
