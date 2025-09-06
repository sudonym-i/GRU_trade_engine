"""
Configuration utilities for TSR data pipelines.
Provides access to config.json settings from the integrations_&_strategy directory.
"""

import json
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_config_path() -> str:
    """
    Get the path to the config.json file in integrations_&_strategy directory.
    
    Returns:
        Absolute path to config.json
    """
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..', '..')
    config_path = os.path.join(project_root, 'integrations_&_strategy', 'config.json')
    return os.path.abspath(config_path)


def load_config() -> dict:
    """
    Load configuration from config.json file.
    
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config.json doesn't exist
        json.JSONDecodeError: If config.json is invalid
    """
    config_path = get_config_path()
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in config file: {e}", e.doc, e.pos)


def get_time_interval() -> str:
    """
    Get the time interval from config.json.
    
    Returns:
        Time interval string (e.g., '1hr', '5min', '1d')
        Defaults to '1d' if not found in config
    """
    try:
        config = load_config()
        interval = config.get('time_interval', '1d')
        logger.info(f"Using time interval: {interval}")
        return interval
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load config: {e}. Using default interval '1d'")
        return '1d'


def get_target_stock() -> Optional[str]:
    """
    Get the target stock ticker from config.json.
    
    Returns:
        Stock ticker string or None if not found
    """
    try:
        config = load_config()
        ticker = config.get('target_stock')
        logger.info(f"Target stock: {ticker}")
        return ticker
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load config: {e}")
        return None


def get_semantic_name() -> Optional[str]:
    """
    Get the semantic name from config.json.
    
    Returns:
        Semantic name string or None if not found
    """
    try:
        config = load_config()
        semantic_name = config.get('semantic_name')
        logger.info(f"Semantic name: {semantic_name}")
        return semantic_name
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load config: {e}")
        return None