# config.py
from pathlib import Path
import yaml
from typing import Dict, Any

DEFAULT_CONFIG = {
    'data': {
        'products_path': 'data/amazon_products.csv',
        'categories_path': 'data/amazon_categories.csv'
    },
    'thresholds': {
        'min_price': 0.0,
        'high_rating': 4.5,
        'high_item_count': 10000,
        'high_volume': 1.06875835e+07
    },
    'amazon_brands': [
        'Pinzon', 'Amazon Basics', 'AmazonBasics', 'Solimo',
        'Amazon Elements', 'AmazonElements', 'Amazon Brand',
        'Mama Bear', 'Wickedly Prime', 'Whole Foods',
        'AmazonFresh', 'Vedaka', 'Goodthreads', '206 Collective',
        'Amazon Essentials', 'AmazonEssentials', 'Core 10'
    ]
}

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file or use defaults
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration values
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Merge with defaults, keeping user values
            return {**DEFAULT_CONFIG, **config}
    except FileNotFoundError:
        return DEFAULT_CONFIG