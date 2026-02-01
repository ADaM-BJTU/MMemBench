"""
Configuration Loader for M3Bench
=================================
Loads and manages dataset configurations from YAML file.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DatasetConfig:
    """Configuration for a single dataset."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize from config dictionary."""
        self.config = config_dict
        self.name = config_dict.get('name', '')
        self.path = Path(config_dict.get('path', ''))
        self.alternate_paths = [Path(p) for p in config_dict.get('alternate_paths', [])]
        self.splits = config_dict.get('splits', [])
        self.annotation_format = config_dict.get('annotation_format', {})
        self.capabilities = config_dict.get('capabilities', {})
        self.supported_tasks = config_dict.get('supported_tasks', [])
        self.task_configs = config_dict.get('task_configs', {})

    def get_annotation_file(self, split: str, file_type: str = 'annotations') -> Path:
        """Get annotation file path for a split."""
        files_config = self.annotation_format.get('files', {})

        # Handle different annotation types
        if file_type in files_config:
            file_pattern = files_config[file_type]
            return self.path / file_pattern.format(split=split)

        return None

    def get_image_path(self, sample_info: Dict[str, Any], split: str = None) -> Path:
        """
        Get image path for a sample.

        Args:
            sample_info: Sample data (may contain img_fn, filename, image_id, etc.)
            split: Data split (train/val/test)
        """
        image_path_pattern = self.annotation_format.get('image_path', '')

        # Try primary path
        if 'img_fn' in sample_info:
            # VCR style
            img_path = self.path / sample_info['img_fn']
            if img_path.exists():
                return img_path

        if 'filename' in sample_info:
            # COCO style
            img_path = self.path / image_path_pattern.format(
                split=split or '',
                filename=sample_info['filename']
            )
            if img_path.exists():
                return img_path

            # Try fallback paths
            for fallback_pattern in self.annotation_format.get('image_fallback_paths', []):
                img_path = self.path / fallback_pattern.format(filename=sample_info['filename'])
                if img_path.exists():
                    return img_path

        if 'image_id' in sample_info:
            # Visual Genome style
            img_path = self.path / image_path_pattern.format(image_id=sample_info['image_id'])
            if img_path.exists():
                return img_path

        # Return pattern-based path even if doesn't exist
        return self.path / image_path_pattern.format(
            split=split or '',
            **sample_info
        )

    def get_task_config(self, task_type: str) -> Dict[str, Any]:
        """Get configuration for a specific task type."""
        return self.task_configs.get(task_type, {})

    def supports_task(self, task_type: str) -> bool:
        """Check if dataset supports a task type."""
        return task_type in self.supported_tasks

    def is_task_enabled(self, task_type: str) -> bool:
        """Check if task generation is enabled for this task type."""
        task_cfg = self.get_task_config(task_type)
        return task_cfg.get('enabled', False)

    def get_capability(self, capability: str) -> bool:
        """Check if dataset has a capability."""
        return self.capabilities.get(capability, False)

    def validate_paths(self) -> bool:
        """Validate that dataset paths exist."""
        if self.path.exists():
            return True

        # Try alternate paths
        for alt_path in self.alternate_paths:
            if alt_path.exists():
                self.path = alt_path
                return True

        return False


class ConfigLoader:
    """Load and manage dataset configurations."""

    def __init__(self, config_file: str = "dataset_configs.yaml"):
        """
        Initialize config loader.

        Args:
            config_file: Path to YAML configuration file
        """
        self.config_file = Path(config_file)
        self.config = {}
        self.datasets = {}
        self.task_templates = {}
        self.quality_control = {}
        self.output_format = {}

        self.load()

    def load(self):
        """Load configuration from YAML file."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        with open(self.config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Parse datasets
        datasets_config = self.config.get('datasets', {})
        for dataset_id, dataset_config in datasets_config.items():
            self.datasets[dataset_id] = DatasetConfig(dataset_config)

        # Parse task templates
        self.task_templates = self.config.get('task_templates', {})

        # Parse quality control rules
        self.quality_control = self.config.get('quality_control', {})

        # Parse output format
        self.output_format = self.config.get('output_format', {})

        logger.info(f"Loaded {len(self.datasets)} dataset configurations")

    def get_dataset_config(self, dataset_id: str) -> Optional[DatasetConfig]:
        """Get configuration for a dataset."""
        return self.datasets.get(dataset_id)

    def get_all_dataset_ids(self) -> List[str]:
        """Get list of all configured dataset IDs."""
        return list(self.datasets.keys())

    def get_datasets_supporting_task(self, task_type: str) -> List[str]:
        """Get list of datasets that support a task type."""
        return [
            dataset_id
            for dataset_id, config in self.datasets.items()
            if config.supports_task(task_type) and config.is_task_enabled(task_type)
        ]

    def get_task_template(self, task_type: str, template_key: str) -> Any:
        """Get a task template."""
        task_templates = self.task_templates.get(task_type, {})
        return task_templates.get(template_key, [])

    def get_question_templates(self, task_type: str, subtype: Optional[str] = None) -> List[str]:
        """Get question templates for a task type."""
        templates = self.get_task_template(task_type, 'question_templates')

        if isinstance(templates, dict) and subtype:
            return templates.get(subtype, [])
        elif isinstance(templates, list):
            return templates

        return []

    def get_answer_template(self, task_type: str) -> str:
        """Get answer template for a task type."""
        return self.get_task_template(task_type, 'answer_template')

    def get_quality_rules(self, task_type: str = None) -> Dict[str, Any]:
        """Get quality control rules."""
        if task_type:
            return self.quality_control.get(task_type, {})
        return self.quality_control.get('global', {})

    def get_output_format(self) -> Dict[str, Any]:
        """Get output format specification."""
        return self.output_format

    def validate_dataset_paths(self) -> Dict[str, bool]:
        """Validate all dataset paths."""
        results = {}
        for dataset_id, config in self.datasets.items():
            results[dataset_id] = config.validate_paths()
            if results[dataset_id]:
                logger.info(f"✓ {dataset_id}: {config.path}")
            else:
                logger.warning(f"✗ {dataset_id}: Path not found")
        return results

    def get_supported_tasks(self) -> List[str]:
        """Get list of all supported task types."""
        tasks = set()
        for config in self.datasets.values():
            tasks.update(config.supported_tasks)
        return sorted(list(tasks))

    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of all configurations."""
        return {
            'total_datasets': len(self.datasets),
            'datasets': {
                dataset_id: {
                    'name': config.name,
                    'path_valid': config.validate_paths(),
                    'splits': config.splits,
                    'supported_tasks': config.supported_tasks,
                    'enabled_tasks': [
                        task for task in config.supported_tasks
                        if config.is_task_enabled(task)
                    ]
                }
                for dataset_id, config in self.datasets.items()
            },
            'supported_tasks': self.get_supported_tasks(),
            'task_templates': list(self.task_templates.keys())
        }


# Convenience function
def load_config(config_file: str = "dataset_configs.yaml") -> ConfigLoader:
    """Load configuration from file."""
    return ConfigLoader(config_file)


if __name__ == "__main__":
    # Test loading
    logging.basicConfig(level=logging.INFO)

    try:
        config = load_config()

        print("\n" + "="*70)
        print("Dataset Configuration Summary")
        print("="*70)

        summary = config.generate_summary()

        print(f"\nTotal Datasets: {summary['total_datasets']}")
        print(f"Supported Tasks: {', '.join(summary['supported_tasks'])}")

        print("\nDataset Status:")
        for dataset_id, info in summary['datasets'].items():
            status = "✓" if info['path_valid'] else "✗"
            print(f"  {status} {dataset_id}: {info['name']}")
            print(f"     Enabled tasks: {', '.join(info['enabled_tasks']) or 'None'}")

        print("\nPath Validation:")
        config.validate_dataset_paths()

    except Exception as e:
        print(f"Error loading config: {e}")
        import traceback
        traceback.print_exc()
