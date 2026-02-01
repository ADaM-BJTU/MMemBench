#!/usr/bin/env python3
"""
快速准备人类标注环境
"""

import os
import shutil
from pathlib import Path


def main():
    print("="*60)
    print("M3Bench 人类评估 - 快速准备")
    print("="*60)
    print()

    # 检查annotation_samples.jsonl是否存在
    samples_file = Path("human_annotations/annotation_samples.jsonl")

    if not samples_file.exists():
        print("❌ 错误: annotation_samples.jsonl 不存在")
        print()
        print("请先运行:")
        print("  python tools/extract_annotation_samples.py")
        print()
        return

    # 检查annotation_results.jsonl是否已存在
    results_file = Path("human_annotations/annotation_results.jsonl")

    if results_file.exists():
        print("⚠️  警告: annotation_results.jsonl 已存在")
        response = input("是否覆盖? (y/n): ")
        if response.lower() != 'y':
            print("取消操作")
            return

    # 复制文件
    print("\n📋 复制标注模板...")
    shutil.copy(samples_file, results_file)
    print(f"✅ 已创建: {results_file}")

    # 统计样本数
    with open(results_file, 'r', encoding='utf-8') as f:
        num_samples = sum(1 for line in f if line.strip())

    print(f"\n📊 样本统计:")
    print(f"  - 总样本数: {num_samples}")
    print(f"  - 预计时间: {num_samples * 2}-{num_samples * 3} 分钟")

    print("\n📚 下一步:")
    print("  1. 阅读标注指南:")
    print("     tools/annotation_guidelines.md")
    print()
    print("  2. 查看标注示例:")
    print("     tools/ANNOTATION_EXAMPLES.md")
    print()
    print("  3. 开始标注:")
    print(f"     编辑 {results_file}")
    print()
    print("  4. 完成后运行验证:")
    print("     python tools/validate_evaluation.py")
    print()

    print("="*60)
    print("祝标注顺利! 🚀")
    print("="*60)


if __name__ == "__main__":
    main()
