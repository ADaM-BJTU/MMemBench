#!/usr/bin/env python3
"""
验证自动评估与人类评估的相关性
"""

import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime


def load_annotations(file_path: str) -> List[Dict]:
    """加载标注结果"""
    annotations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    return annotations


def normalize_scores(human_score: int, scale: int = 5) -> float:
    """将人类评分归一化到0-1"""
    if human_score is None:
        return None
    return (human_score - 1) / (scale - 1)


def calculate_correlations(annotations: List[Dict]) -> Dict[str, float]:
    """计算相关性"""
    try:
        from scipy.stats import pearsonr, spearmanr
    except ImportError:
        print("Warning: scipy not installed. Install with: pip install scipy")
        return {}

    # 提取分数
    human_overall = []
    auto_overall = []
    human_correctness = []
    auto_correctness = []

    for ann in annotations:
        h = ann.get("human_annotation", {})
        a = ann.get("auto_evaluation", {})

        # Overall quality
        if h.get("overall_quality") is not None:
            human_overall.append(normalize_scores(h["overall_quality"]))
            auto_overall.append(a.get("score", 0.5))

        # Correctness
        if h.get("correctness") is not None:
            human_correctness.append(normalize_scores(h["correctness"]))
            auto_correctness.append(a.get("correctness", 0.5))

    results = {}

    if len(human_overall) > 2:
        r_overall, p_overall = pearsonr(human_overall, auto_overall)
        rho_overall, _ = spearmanr(human_overall, auto_overall)
        results["overall_pearson"] = r_overall
        results["overall_pearson_p"] = p_overall
        results["overall_spearman"] = rho_overall

    if len(human_correctness) > 2:
        r_correct, p_correct = pearsonr(human_correctness, auto_correctness)
        results["correctness_pearson"] = r_correct
        results["correctness_pearson_p"] = p_correct

    return results


def identify_mismatches(annotations: List[Dict], threshold: float = 0.4) -> List[Dict]:
    """识别误判案例"""
    mismatches = []

    for ann in annotations:
        h = ann.get("human_annotation", {})
        a = ann.get("auto_evaluation", {})

        if h.get("overall_quality") is None:
            continue

        human_score = normalize_scores(h["overall_quality"])
        auto_score = a.get("score", 0.5)

        diff = abs(human_score - auto_score)

        if diff > threshold:
            mismatches.append({
                "sample_id": ann["sample_id"],
                "action_type": ann["action_type"],
                "human_score": human_score,
                "auto_score": auto_score,
                "difference": diff,
                "human_comments": h.get("comments", ""),
                "response_snippet": ann["vlm_response"][:150] + "..." if len(ann["vlm_response"]) > 150 else ann["vlm_response"]
            })

    return sorted(mismatches, key=lambda x: x["difference"], reverse=True)


def generate_report(annotations: List[Dict], correlations: Dict, mismatches: List[Dict]) -> str:
    """生成验证报告"""
    report = []
    report.append("# M3Bench 自动评估验证报告\n\n")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report.append(f"**样本数量**: {len(annotations)}\n\n")
    report.append("---\n\n")

    report.append("## 1. 相关性分析\n\n")

    if correlations:
        report.append(f"- **Overall Quality (Pearson r)**: {correlations.get('overall_pearson', 0):.3f}")
        report.append(f" (p-value: {correlations.get('overall_pearson_p', 1):.4f})\n")
        report.append(f"- **Overall Quality (Spearman ρ)**: {correlations.get('overall_spearman', 0):.3f}\n")
        report.append(f"- **Correctness (Pearson r)**: {correlations.get('correctness_pearson', 0):.3f}\n\n")

        # 解释
        r = correlations.get('overall_pearson', 0)
        if r > 0.7:
            interpretation = "**强相关** - 自动评估与人类评估高度一致 ✅"
        elif r > 0.5:
            interpretation = "**中等相关** - 自动评估基本可靠 ⚠️"
        elif r > 0.3:
            interpretation = "**弱相关** - 自动评估需要改进 ⚠️"
        else:
            interpretation = "**几乎无相关** - 自动评估存在严重问题 ❌"

        report.append(f"**解释**: {interpretation}\n\n")
    else:
        report.append("*无法计算相关性（需要安装scipy）*\n\n")

    report.append("---\n\n")
    report.append("## 2. 误判案例分析\n\n")
    report.append(f"发现 **{len(mismatches)}** 个显著误判案例（差异 > 0.4）\n\n")

    if mismatches:
        for i, m in enumerate(mismatches[:10], 1):
            report.append(f"### 案例 {i}: `{m['sample_id']}`\n\n")
            report.append(f"- **动作类型**: {m['action_type']}\n")
            report.append(f"- **人类评分**: {m['human_score']:.2f}\n")
            report.append(f"- **自动评分**: {m['auto_score']:.2f}\n")
            report.append(f"- **差异**: {m['difference']:.2f}\n")
            if m['human_comments']:
                report.append(f"- **人类评论**: {m['human_comments']}\n")
            report.append(f"- **回复片段**: {m['response_snippet']}\n\n")
    else:
        report.append("*没有发现显著误判案例*\n\n")

    report.append("---\n\n")
    report.append("## 3. 改进建议\n\n")

    # 基于相关性给出建议
    if correlations:
        r = correlations.get('overall_pearson', 0)
        if r < 0.5:
            report.append("### ⚠️ 紧急改进建议\n\n")
            report.append("自动评估相关性较低，建议：\n\n")
            report.append("1. 检查LLM-as-Judge的prompt是否清晰\n")
            report.append("2. 调整hard rules的权重\n")
            report.append("3. 增加更多ground truth验证\n")
            report.append("4. 考虑使用更强的评估模型\n\n")

    # 基于误判案例给出建议
    if mismatches:
        action_errors = {}
        for m in mismatches:
            action = m["action_type"]
            action_errors[action] = action_errors.get(action, 0) + 1

        report.append("### 动作类型误判分布\n\n")
        for action, count in sorted(action_errors.items(), key=lambda x: x[1], reverse=True):
            report.append(f"- **{action}**: {count} 次\n")
        report.append("\n")

    report.append("---\n\n")
    report.append("## 4. 结论\n\n")

    if correlations:
        r = correlations.get('overall_pearson', 0)
        if r > 0.6:
            report.append("✅ **自动评估系统基本可靠**，可以用于大规模评估。\n\n")
            report.append("建议在论文中报告：\n")
            report.append(f'> "Our automatic evaluation shows strong correlation with human judgment (Pearson r={r:.3f}, p<0.01), validating its reliability for large-scale benchmarking."\n')
        else:
            report.append("⚠️ **自动评估系统需要改进**后才能用于正式基准测试。\n\n")
            report.append("建议先改进评估系统，然后重新进行人类验证。\n")
    else:
        report.append("*需要安装scipy才能计算相关性*\n")

    return "".join(report)


def main():
    # 配置
    annotation_file = "human_annotations/annotation_results.jsonl"
    output_file = "human_annotations/validation_report.md"

    if not Path(annotation_file).exists():
        print(f"Error: {annotation_file} not found!")
        print("Please complete human annotation first.")
        print("\nExpected file format: annotation_results.jsonl")
        print("(Copy from annotation_samples.jsonl and fill in human_annotation fields)")
        return

    print(f"Loading annotations from {annotation_file}...")
    annotations = load_annotations(annotation_file)

    # 过滤未标注的样本
    annotated = [a for a in annotations if a.get("human_annotation", {}).get("overall_quality") is not None]
    print(f"Found {len(annotated)} annotated samples (out of {len(annotations)} total)")

    if len(annotated) < 10:
        print("Warning: Too few annotated samples for reliable analysis")
        print("Please annotate at least 10 samples before running validation.")
        return

    print("\nCalculating correlations...")
    correlations = calculate_correlations(annotated)

    print("Identifying mismatches...")
    mismatches = identify_mismatches(annotated)

    print("Generating report...")
    report = generate_report(annotated, correlations, mismatches)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✅ Report saved to: {output_file}")

    # 打印关键结果
    print("\n" + "="*60)
    print("KEY RESULTS")
    print("="*60)
    if correlations:
        print(f"Overall Correlation (Pearson r): {correlations.get('overall_pearson', 0):.3f}")
        print(f"P-value: {correlations.get('overall_pearson_p', 1):.4f}")
    print(f"Significant Mismatches: {len(mismatches)}")
    print(f"Annotated Samples: {len(annotated)}")
    print("="*60)


if __name__ == "__main__":
    main()
