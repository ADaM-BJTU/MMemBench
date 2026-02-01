"""
Example Usage of M3Bench DataProvider
=====================================

This script demonstrates how to use DataLoader and DataGeneratorV2.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataprovider import DataLoader, DataGeneratorV2, load_config
import json


def example_1_load_mscoco():
    """Example 1: Load MSCOCO dataset."""
    print("=" * 60)
    print("Example 1: Load MSCOCO14 Dataset")
    print("=" * 60)

    loader = DataLoader(data_root="data")

    # Load dataset
    print("\nLoading MSCOCO14 train split (max 5 samples)...")
    data = loader.load_dataset(
        dataset_id="mscoco14",
        split="train",
        max_samples=5
    )

    # Display first sample
    if data:
        sample = data[0]
        print(f"\nSample ID: {sample['sample_id']}")
        print(f"Image: {sample['file_name']}")
        print(f"Dimensions: {sample['width']}x{sample['height']}")
        print(f"Number of objects: {len(sample['objects'])}")
        print(f"Categories: {[obj['category'] for obj in sample['objects'][:5]]}")
        print(f"Number of captions: {len(sample['captions'])}")
        if sample['captions']:
            print(f"First caption: {sample['captions'][0]}")
    else:
        print("No data loaded. Make sure MSCOCO14 data exists in data/MSCOCO14/example/")

    print()


def example_2_load_vcr():
    """Example 2: Load VCR dataset."""
    print("=" * 60)
    print("Example 2: Load VCR (Visual Commonsense Reasoning) Dataset")
    print("=" * 60)

    loader = DataLoader(data_root="data")

    # Load dataset
    print("\nLoading VCR train split (max 3 samples)...")
    data = loader.load_dataset(
        dataset_id="vcr",
        split="train",
        max_samples=3
    )

    # Display first sample
    if data:
        sample = data[0]
        print(f"\nSample ID: {sample['sample_id']}")
        print(f"Image: {Path(sample['image_path']).name}")
        print(f"Objects: {sample['objects']}")
        print(f"Question tokens: {sample['question']}")
        print(f"Answer choices: {len(sample['answer_choices'])} choices")
        print(f"Correct answer: {sample['answer_label']}")
    else:
        print("No data loaded. Make sure VCR data exists in data/VisualCommenReasoning/")

    print()


def example_3_generate_vnf():
    """Example 3: Generate Visual Noise Filtering tasks."""
    print("=" * 60)
    print("Example 3: Generate Visual Noise Filtering Tasks (V2)")
    print("=" * 60)

    loader = DataLoader(data_root="data")
    config = load_config("dataset_configs.yaml")
    generator = DataGeneratorV2(loader, config)

    # Generate tasks
    print("\nGenerating 3 VNF tasks from MSCOCO14...")
    try:
        tasks = generator.generate_tasks(
            dataset_id="mscoco14",
            task_type="VNF",
            num_tasks=3,
            split="train"
        )

        if tasks:
            print(f"Generated {len(tasks)} tasks!\n")

            # Display first task
            task = tasks[0]
            print(f"Task ID: {task['task_id']}")
            print(f"Task Type: {task['task_type']}")
            print(f"Question: {task['question']}")
            print(f"Number of images: {len(task['images'])}")
            print(f"Target image index: {task['target_image_idx']}")
            print(f"Answer: {task['answer']}")
            print(f"Reasoning depth: {task['reasoning_depth']}")
            print(f"\nMetadata:")
            for key, value in task['metadata'].items():
                print(f"  {key}: {value}")

            # Save tasks
            output_path = Path("output/vnf_example.jsonl")
            generator.save_generated_tasks(tasks, str(output_path))
            print(f"\nSaved tasks to: {output_path}")
        else:
            print("No tasks generated. Make sure MSCOCO14 data exists.")

    except Exception as e:
        print(f"Error generating tasks: {e}")

    print()


def example_4_generate_abr():
    """Example 4: Generate Attribute-Bridge Reasoning tasks."""
    print("=" * 60)
    print("Example 4: Generate Attribute-Bridge Multi-hop Reasoning Tasks (V2)")
    print("=" * 60)

    loader = DataLoader(data_root="data")
    config = load_config("dataset_configs.yaml")
    generator = DataGeneratorV2(loader, config)

    # Generate tasks from Visual Genome
    print("\nGenerating 2 ABR tasks from Visual Genome...")
    try:
        tasks = generator.generate_tasks(
            dataset_id="visual_genome",
            task_type="ABR",
            num_tasks=2,
            split="train"
        )

        if tasks:
            print(f"Generated {len(tasks)} tasks!\n")

            # Display first task
            task = tasks[0]
            print(f"Task ID: {task['task_id']}")
            print(f"Task Type: {task['task_type']}")
            print(f"Question: {task['question']}")
            print(f"Answer: {task['answer']}")
            print(f"Reasoning depth: {task['reasoning_depth']} hops")

            print(f"\nFormal Representation:")
            print(f"Objects (O): {task['formal_representation']['objects']}")
            print(f"Attributes (A): {json.dumps(task['formal_representation']['attributes'], indent=2)}")
            print(f"Relations (R): {json.dumps(task['formal_representation']['relations'], indent=2)}")

            if task['reasoning_path']:
                print(f"\nReasoning Path:")
                for step in task['reasoning_path']:
                    from_obj = step['from_object']
                    to_obj = step['to_object']
                    print(f"  Hop {step['hop']}: {from_obj['name']} "
                          f"(attrs: {from_obj['attributes']}) "
                          f"--[{step['relation']}]--> "
                          f"{to_obj['name']} (attrs: {to_obj['attributes']})")

            # Save tasks
            output_path = Path("output/abr_example.jsonl")
            generator.save_generated_tasks(tasks, str(output_path))
            print(f"\nSaved tasks to: {output_path}")
        else:
            print("No tasks generated. Make sure Visual Genome data exists.")

    except Exception as e:
        print(f"Error generating tasks: {e}")

    print()


def example_5_generate_rc():
    """Example 5: Generate Relation Comparison tasks."""
    print("=" * 60)
    print("Example 5: Generate Relation Comparison Tasks (V2)")
    print("=" * 60)

    loader = DataLoader(data_root="data")
    config = load_config("dataset_configs.yaml")
    generator = DataGeneratorV2(loader, config)

    # Generate tasks
    print("\nGenerating 3 Relation Comparison tasks from MSCOCO14...")
    try:
        tasks = generator.generate_tasks(
            dataset_id="mscoco14",
            task_type="RC",
            num_tasks=3,
            split="train"
        )

        if tasks:
            print(f"Generated {len(tasks)} tasks!\n")

            # Display first task
            task = tasks[0]
            print(f"Task ID: {task['task_id']}")
            print(f"Task Type: {task['task_type']}")
            print(f"Question: {task['question']}")
            print(f"Number of images: {len(task['images'])}")
            print(f"Answer: {task['answer']}")
            print(f"Comparison target: {task['comparison_target']}")
            print(f"Reasoning depth: {task['reasoning_depth']}")

            if 'counts' in task['metadata']:
                print(f"\nCounts per image: {task['metadata']['counts']}")

            # Save tasks
            output_path = Path("output/rc_example.jsonl")
            generator.save_generated_tasks(tasks, str(output_path))
            print(f"\nSaved tasks to: {output_path}")
        else:
            print("No tasks generated. Make sure MSCOCO14 data exists.")

    except Exception as e:
        print(f"Error generating tasks: {e}")

    print()


def example_6_generate_ac():
    """Example 6: Generate Attribute Comparison tasks."""
    print("=" * 60)
    print("Example 6: Generate Attribute Comparison Tasks (V2)")
    print("=" * 60)

    loader = DataLoader(data_root="data")
    config = load_config("dataset_configs.yaml")
    generator = DataGeneratorV2(loader, config)

    # Generate tasks
    print("\nGenerating 2 Attribute Comparison tasks from MSCOCO14...")
    try:
        tasks = generator.generate_tasks(
            dataset_id="mscoco14",
            task_type="AC",
            num_tasks=2,
            split="train"
        )

        if tasks:
            print(f"Generated {len(tasks)} tasks!\n")

            # Display first task
            task = tasks[0]
            print(f"Task ID: {task['task_id']}")
            print(f"Task Type: {task['task_type']}")
            print(f"Question: {task['question']}")
            print(f"Number of images: {len(task['images'])}")
            print(f"Answer: {task['answer']}")

            # Save tasks
            output_path = Path("output/ac_example.jsonl")
            with open(output_path, 'w', encoding='utf-8') as f:
                for task in tasks:
                    f.write(json.dumps(task, ensure_ascii=False) + '\n')
            print(f"\nSaved tasks to: {output_path}")
        else:
            print("No tasks generated. Make sure MSCOCO14 data exists.")

    except Exception as e:
        print(f"Error generating tasks: {e}")

    print()


def example_7_dataset_info():
    """Example 7: Get dataset information."""
    print("=" * 60)
    print("Example 7: Get Dataset Information")
    print("=" * 60)

    loader = DataLoader(data_root="data")

    datasets = ["mscoco14", "visual_genome", "vcr", "mm-niah"]

    for dataset_id in datasets:
        info = loader.get_dataset_info(dataset_id)

        if info:
            print(f"\nDataset: {info['name']}")
            print(f"Description: {info['description']}")
            print(f"Splits: {info['splits']}")
            print(f"Modalities: {info['modalities']}")
            print(f"Task Types: {info['task_types']}")

    print()


def example_8_supported_tasks():
    """Example 8: Check supported task types."""
    print("=" * 60)
    print("Example 8: Check Supported Task Types (V2)")
    print("=" * 60)

    config = load_config("dataset_configs.yaml")

    datasets = ["mscoco14", "visual_genome", "vcr", "sherlock"]

    print("\nDataset-Task Compatibility:\n")

    for dataset_id in datasets:
        if dataset_id in config:
            dataset_config = config[dataset_id]
            supported = dataset_config.get('supported_task_types', [])
            print(f"{dataset_id}:")
            if supported:
                for task_type in supported:
                    print(f"  ✓ {task_type}")
            else:
                print(f"  (no task generation support)")
        else:
            print(f"{dataset_id}: (not configured)")
        print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("M3Bench DataProvider Examples (V2)")
    print("=" * 60 + "\n")

    # Create output directory
    Path("output").mkdir(exist_ok=True)

    # Run examples
    examples = [
        example_1_load_mscoco,
        example_2_load_vcr,
        example_3_generate_vnf,
        example_4_generate_abr,
        example_5_generate_rc,
        example_6_generate_ac,
        example_7_dataset_info,
        example_8_supported_tasks,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}\n")

    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()