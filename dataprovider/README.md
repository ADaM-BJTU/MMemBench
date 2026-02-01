# M3Bench DataProvider

Complete data loading and generation module for M3Bench benchmark.

## Architecture

```
dataprovider/
├── __init__.py         # Package entry point
├── loader.py           # DataLoader: Load datasets by ID
├── generator.py        # DataGenerator: Synthesize new test datasets
├── filters.py          # Data filtering utilities (future)
└── README.md          # This file
```

## Features

### DataLoader

Load and parse various multimodal datasets with unified format:

**Supported Datasets:**
- **MSCOCO14**: Object detection with bounding boxes and captions
- **Visual Genome**: Scene graphs with objects, attributes, relationships
- **VCR**: Visual Commonsense Reasoning with Q-A-R
- **Sherlock**: Clue-inference pairs with visual grounding
- **MM-NIAH**: Needle in multimodal haystack (retrieval/counting/reasoning)
- **MMMU**: Massive multi-discipline multimodal understanding
- **MMDocIR**: Long document multi-modal retrieval
- **ScienceQA, RealworldQA, DocVQA, InfoGraphVQA, SlideVQA, MMStar, GQA**

### DataGenerator

Synthesize new test datasets using 4 task types:

1. **Visual Noise Filtering (VNF)**
   - N images + 1 question
   - Only 1 image contains answer
   - Tests retrieval capability

2. **Attribute-Bridge Multi-hop Reasoning (ABR)**
   - Chain reasoning through attributes and relationships
   - Formal representation: O={o1,o2,...}, A(oi)={...}, R={...}
   - Tests reasoning depth

3. **Relation Comparison Reasoning (RC)**
   - Compare similar objects across multiple images
   - Tests aggregation and comparison

4. **Symbol Memory (SM)**
   - Associate icons with intents/workflows
   - Tests context management

## Quick Start

### Installation

```bash
cd M3Bench_new
pip install -r requirements.txt  # jsonlines, Pillow, etc.
```

### Basic Usage

```python
from dataprovider import DataLoader, DataGenerator

# Initialize
loader = DataLoader(data_root="data")
generator = DataGenerator(loader)

# Load a dataset
mscoco_data = loader.load_dataset(
    dataset_id="mscoco14",
    split="train",
    max_samples=100
)

# Generate Visual Noise Filtering tasks
vnf_tasks = generator.generate_task(
    task_type="visual_noise_filtering",
    source_dataset="mscoco14",
    num_samples=50,
    n_distractor_images=3
)

# Save generated tasks
generator.save_generated_tasks(
    vnf_tasks,
    output_path="output/vnf_tasks.jsonl",
    format="jsonl"
)
```

## Detailed Examples

### Example 1: Load MSCOCO and Filter

```python
from dataprovider import DataLoader

loader = DataLoader("data")

# Load MSCOCO14
data = loader.load_dataset("mscoco14", split="train")

# Filter samples with 2-5 objects
filtered = loader.filter_by_attributes(
    data,
    filters={
        'num_objects': (2, 5),  # Range filter
    }
)

# Access data
for sample in filtered[:5]:
    print(f"Image: {sample['image_path']}")
    print(f"Objects: {[obj['category'] for obj in sample['objects']]}")
    print(f"Captions: {sample['captions']}")
    print()
```

**Output Format:**
```python
{
    'dataset_id': 'mscoco14',
    'sample_id': 'mscoco14_123456',
    'image_path': 'data/MSCOCO14/train2014/COCO_train2014_000000123456.jpg',
    'image_id': 123456,
    'objects': [
        {
            'id': 789,
            'category': 'person',
            'category_id': 1,
            'bbox': [100, 150, 80, 200],  # [x, y, width, height]
            'area': 16000,
            'segmentation': [...],
            'iscrowd': 0
        },
        # ... more objects
    ],
    'captions': [
        'A person standing in a room',
        'Someone is standing indoors'
    ],
    'width': 640,
    'height': 480
}
```

### Example 2: Generate Visual Noise Filtering Tasks

```python
from dataprovider import DataLoader, DataGenerator

loader = DataLoader("data")
generator = DataGenerator(loader)

# Generate from MSCOCO
vnf_tasks = generator.generate_task(
    task_type="visual_noise_filtering",
    source_dataset="mscoco14",
    num_samples=100,
    n_distractor_images=4  # 4 distractors + 1 target = 5 images total
)

# Inspect a task
task = vnf_tasks[0]
print(f"Task ID: {task['task_id']}")
print(f"Question: {task['question']}")
print(f"Images: {len(task['images'])} images")
print(f"Target: Image {task['target_image_idx']}")
print(f"Answer: {task['answer']}")
```

**Output Format:**
```python
{
    'task_id': 'vnf_mscoco_0',
    'task_type': 'visual_noise_filtering',
    'images': [
        'data/MSCOCO14/train2014/image1.jpg',
        'data/MSCOCO14/train2014/image2.jpg',  # Target image
        'data/MSCOCO14/train2014/image3.jpg',
        'data/MSCOCO14/train2014/image4.jpg',
        'data/MSCOCO14/train2014/image5.jpg'
    ],
    'question': 'Which image contains a dog?',
    'target_image_idx': 1,
    'answer': 'Image 1',
    'reasoning_depth': 1,
    'metadata': {
        'source_dataset': 'mscoco14',
        'n_total_images': 5,
        'target_category': 'dog',
        'distractor_strategy': 'non_overlapping_categories'
    }
}
```

### Example 3: Generate Attribute-Bridge Multi-hop Reasoning

```python
from dataprovider import DataLoader, DataGenerator

loader = DataLoader("data")
generator = DataGenerator(loader)

# Generate from Visual Genome
abr_tasks = generator.generate_task(
    task_type="attribute_bridge_reasoning",
    source_dataset="visual_genome",
    num_samples=50,
    min_hops=2,
    max_hops=4
)

# Inspect a task
task = abr_tasks[0]
print(f"Task ID: {task['task_id']}")
print(f"Question: {task['question']}")
print(f"Reasoning Depth: {task['reasoning_depth']} hops")
print(f"\nFormal Representation:")
print(f"Objects (O): {task['formal_representation']['objects']}")
print(f"Attributes (A): {task['formal_representation']['attributes']}")
print(f"Relations (R): {task['formal_representation']['relations']}")
print(f"\nReasoning Path:")
for step in task['reasoning_path']:
    print(f"  Hop {step['hop']}: {step['from_object']['name']} "
          f"--[{step['relation']}]--> {step['to_object']['name']}")
```

**Output Format:**
```python
{
    'task_id': 'abr_vg_0',
    'task_type': 'attribute_bridge_reasoning',
    'images': ['data/visual_genome/images/123.jpg'],
    'question': "Given object 'person' (attributes: tall, red_shirt), "
                "what is the final object after following the relationship chain?",
    'answer': "The final object is 'car' with attributes: blue, sedan",
    'reasoning_path': [
        {
            'hop': 1,
            'from_object': {
                'id': 'o1',
                'name': 'person',
                'attributes': ['tall', 'red_shirt']
            },
            'relation': 'holding',
            'to_object': {
                'id': 'o2',
                'name': 'umbrella',
                'attributes': ['black', 'open']
            }
        },
        {
            'hop': 2,
            'from_object': {
                'id': 'o2',
                'name': 'umbrella',
                'attributes': ['black', 'open']
            },
            'relation': 'above',
            'to_object': {
                'id': 'o3',
                'name': 'car',
                'attributes': ['blue', 'sedan']
            }
        }
    ],
    'reasoning_depth': 2,
    'formal_representation': {
        'objects': {
            'o1': 'person',
            'o2': 'umbrella',
            'o3': 'car'
        },
        'attributes': {
            'o1': ['tall', 'red_shirt'],
            'o2': ['black', 'open'],
            'o3': ['blue', 'sedan']
        },
        'relations': [
            {'subject': 'o1', 'predicate': 'holding', 'object': 'o2'},
            {'subject': 'o2', 'predicate': 'above', 'object': 'o3'}
        ]
    },
    'metadata': {
        'source_dataset': 'visual_genome',
        'image_id': 123,
        'chain_length': 3
    }
}
```

### Example 4: Generate Relation Comparison Tasks

```python
from dataprovider import DataLoader, DataGenerator

loader = DataLoader("data")
generator = DataGenerator(loader)

# Generate from MSCOCO
rc_tasks = generator.generate_task(
    task_type="relation_comparison",
    source_dataset="mscoco14",
    num_samples=50,
    n_images=4  # Compare across 4 images
)

# Inspect a task
task = rc_tasks[0]
print(f"Task ID: {task['task_id']}")
print(f"Question: {task['question']}")
print(f"Comparison Target: {task['comparison_target']}")
print(f"Number of Images: {len(task['images'])}")
print(f"Answer: {task['answer']}")
```

**Output Format:**
```python
{
    'task_id': 'rc_mscoco_0',
    'task_type': 'relation_comparison',
    'images': [
        'data/MSCOCO14/train2014/image1.jpg',
        'data/MSCOCO14/train2014/image2.jpg',
        'data/MSCOCO14/train2014/image3.jpg',
        'data/MSCOCO14/train2014/image4.jpg'
    ],
    'question': 'Which image has the most dogs?',
    'answer': 'Image 2 has 3 dog(s)',
    'comparison_target': 'count_dog',
    'reasoning_depth': 2,
    'metadata': {
        'source_dataset': 'mscoco14',
        'category': 'dog',
        'counts': [1, 3, 0, 2]
    }
}
```

### Example 5: Generate Symbol Memory Tasks

```python
from dataprovider import DataLoader, DataGenerator

loader = DataLoader("data")
generator = DataGenerator(loader)

# Generate from MSCOCO
sm_tasks = generator.generate_task(
    task_type="symbol_memory",
    source_dataset="mscoco14",
    num_samples=50,
    n_symbols=5  # Associate 5 symbols with intents
)

# Inspect a task
task = sm_tasks[0]
print(f"Task ID: {task['task_id']}")
print(f"\nSymbol Mappings:")
for mapping in task['symbol_mappings']:
    print(f"  {mapping['symbol_name']}: {mapping['intent']}")
print(f"\nQuestion: {task['question']}")
print(f"Answer: {task['answer']}")
```

**Output Format:**
```python
{
    'task_id': 'sm_mscoco_0',
    'task_type': 'symbol_memory',
    'symbol_mappings': [
        {
            'symbol_image': 'data/MSCOCO14/train2014/image1.jpg',
            'symbol_name': 'traffic light',
            'intent': 'control_traffic_flow',
            'description': 'A traffic light represents control traffic flow'
        },
        {
            'symbol_image': 'data/MSCOCO14/train2014/image2.jpg',
            'symbol_name': 'stop sign',
            'intent': 'require_stop',
            'description': 'A stop sign represents require stop'
        },
        # ... more mappings
    ],
    'test_symbols': [
        'data/MSCOCO14/train2014/image1.jpg',
        'data/MSCOCO14/train2014/image2.jpg',
        # ... more images
    ],
    'question': 'What intent does a traffic light represent?',
    'answer': 'control traffic flow',
    'reasoning_depth': 5,  # Number of symbols to remember
    'metadata': {
        'source_dataset': 'mscoco14',
        'n_symbols': 5,
        'test_symbol': 'traffic light'
    }
}
```

## Advanced Features

### Dataset Information

```python
from dataprovider import DataLoader

loader = DataLoader("data")

# Get dataset info
info = loader.get_dataset_info("mscoco14")
print(f"Name: {info['name']}")
print(f"Description: {info['description']}")
print(f"Splits: {info['splits']}")
print(f"Modalities: {info['modalities']}")
print(f"Task Types: {info['task_types']}")
```

### Check Supported Tasks

```python
from dataprovider import DataLoader, DataGenerator

loader = DataLoader("data")
generator = DataGenerator(loader)

# Check what tasks can be generated from a dataset
supported = generator.get_supported_tasks("mscoco14")
print(f"MSCOCO14 supports: {supported}")
# Output: ['visual_noise_filtering', 'attribute_bridge_reasoning',
#          'relation_comparison', 'symbol_memory']

supported = generator.get_supported_tasks("visual_genome")
print(f"Visual Genome supports: {supported}")
# Output: ['attribute_bridge_reasoning', 'relation_comparison']
```

### Batch Generation

```python
from dataprovider import DataLoader, DataGenerator

loader = DataLoader("data")
generator = DataGenerator(loader)

# Generate all task types from MSCOCO
all_tasks = []

for task_type in generator.get_supported_tasks("mscoco14"):
    tasks = generator.generate_task(
        task_type=task_type,
        source_dataset="mscoco14",
        num_samples=25
    )
    all_tasks.extend(tasks)
    print(f"Generated {len(tasks)} {task_type} tasks")

# Save all tasks
generator.save_generated_tasks(
    all_tasks,
    output_path="output/all_mscoco_tasks.jsonl"
)

print(f"\nTotal generated: {len(all_tasks)} tasks")
```

## Data Filtering

```python
from dataprovider import DataLoader

loader = DataLoader("data")

# Load Visual Genome
data = loader.load_dataset("visual_genome", split="train")

# Complex filtering
filtered = loader.filter_by_attributes(
    data,
    filters={
        'scene_graph.num_objects': (3, 10),  # 3-10 objects
        'scene_graph.num_relationships': (2, 5),  # 2-5 relationships
    }
)

print(f"Filtered {len(data)} → {len(filtered)} samples")
```

## Configuration

### Dataset Paths

By default, DataLoader expects datasets in:
```
data/
├── MSCOCO14/
│   └── example/
│       ├── instances_train2014.json
│       ├── captions_train2014.json
│       └── images/
├── visual_genome/
│   ├── scene_graphs_train.json
│   └── images/
├── VisualCommenReasoning/
│   ├── train.jsonl
│   └── images/
└── ...
```

Customize the root path:
```python
loader = DataLoader(data_root="/custom/path/to/data")
```

## Task Type Mapping

| Task Type | Capability Level | Reasoning Depth | Multi-Image |
|-----------|-----------------|-----------------|-------------|
| Visual Noise Filtering | Retrieval | 1 | ✓ |
| Attribute-Bridge Reasoning | Reasoning | 2-4 | ✗ |
| Relation Comparison | Aggregation | 2 | ✓ |
| Symbol Memory | Context Management | Variable | ✗ |

## Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

from dataprovider import DataLoader, DataGenerator

loader = DataLoader("data")
generator = DataGenerator(loader)

# Now all operations will log detailed information
```

## Dataset-Task Compatibility Matrix

| Dataset | VNF | ABR | RC | SM |
|---------|-----|-----|----|----|
| MSCOCO14 | ✓ | ✓ | ✓ | ✓ |
| Visual Genome | ✗ | ✓ | ✓ | ✗ |
| VCR | ✓ | ✓ | ✗ | ✗ |
| Sherlock | ✓ | ✗ | ✗ | ✗ |
| MM-NIAH | ✗ | ✗ | ✗ | ✗ |
| MMMU | ✗ | ✗ | ✗ | ✗ |
| MMDocIR | ✗ | ✗ | ✗ | ✗ |

## Contributing

To add support for a new dataset:

1. Add loader method in `loader.py`:
   ```python
   def _load_your_dataset(self, split: str = "train", **kwargs) -> List[Dict]:
       """Load your dataset."""
       # Implementation
       return samples
   ```

2. Update `SUPPORTED_DATASETS` list

3. Add dataset info in `get_dataset_info()`

4. Implement generation logic in `generator.py` if applicable

## Citation

If you use this dataprovider in your research, please cite:

```bibtex
@misc{m3bench2025,
  title={M3Bench: A Comprehensive Multimodal Benchmark},
  author={Your Name},
  year={2025}
}
```

## License

Apache 2.0

## Contact

For questions or issues, please open an issue on GitHub.