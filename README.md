# M3Bench

M3Bench 是一个用于评估视觉语言模型（VLM）记忆机制的推理与上下文管理能力的动态测试基准框架与评估系统。旨在解决推理能力测试中的全局信息作弊问题，以及上下文管理能力测试中缺少状态管理的问题。

## 项目结构

```
M3Bench_new/
├── dataprovider/           # 数据加载与任务生成
│   ├── loader.py           # 数据集加载器
│   └── generator.py        # 任务生成器
├── src/
│   └── simulator/          # LLM驱动的用户模拟器
│       ├── llm_client.py       # API客户端
│       ├── llm_user_simulator.py  # 核心模拟器
│       ├── memory_store.py     # 对话记忆管理
│       └── task_config.py      # 任务配置与提示词
├── data/                   # 数据集配置
├── generated_tasks_v2/     # 生成的测试任务
└── simulator_test_log/     # 模拟器运行日志
```

## 目前支持的任务类型

| 任务类型 | 英文缩写 | 说明 |
|---------|---------|------|
| 属性比较 | AC (Attribute Comparison) | 跨图像比较对象属性 |
| 视觉噪声过滤 | VNF (Visual Noise Filtering) | 从干扰信息中提取目标信息 |
| 属性桥接推理 | ABR (Attribute Bridge Reasoning) | 多跳空间推理 |
| 关系比较 | RC (Relation Comparison) | 跨图像比较对象关系 |

## 目前支持的数据集

- MSCOCO 2014
- VCR (Visual Commonsense Reasoning)
- ScienceQA
- DocVQA
- RealworldQA

## 快速开始

### 1. 生成测试任务

```bash
python generate_unified.py
```

生成的任务保存在 `generated_tasks_v2/` 目录下。

### 2. 运行用户模拟器

```bash
python test_user_simulator.py
```

模拟器会：
- 加载生成的任务
- 使用核心模型驱动多轮对话
- 测试目标VLM模型
- 记录详细日志到 `simulator_test_log/`

### 3. 查看生成的任务

```bash
python view_tasks.py
```

## 用户模拟器

系统使用LLM驱动的用户模拟器来测试VLM：

- 核心模型：负责选择动作、生成消息、评估回复
- 目标模型系统：被测试的VLM，支持接入记忆机制
- 记忆系统：存储关键信息和对话历史

动作空间：
- guidance: 引导模型关注特定区域
- logic skip : 跳过复杂推理
- redundancy： 提供重复描述和相同信息
- follow_up: 追问细节
- mislead: 注入错误信息测试鲁棒性
- distraction: 注入干扰信息
- fine_grained: 要求精确细节
- negation: 纠正错误推理
- next_task: 结束当前任务

## 配置

API配置在 `src/simulator/llm_client.py` 中设置。

数据集路径在 `dataprovider/loader.py` 中配置。

## 输出文件

任务生成：
- `{task_type}_{dataset}.jsonl` - 生成的任务文件

模拟器日志：
- `run_log_*.json` - 详细运行日志
- `memory_state_*.json` - 记忆状态
- `summary_*.json` - 运行摘要
- `readable_summary.txt` - 可读摘要

## 依赖

- Python 3.8+
- requests
- jsonlines
- pycocotools（MSCOCO数据集）

## 许可

仅供研究使用。
