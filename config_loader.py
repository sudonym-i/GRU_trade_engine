"""
Configuration loader utility for GRU Trade Engine.
Loads and validates configuration from config.json file.
"""

import json
import os
from typing import Dict, Any


class ConfigLoader:
    """Loads and provides access to configuration from config.json"""

    def __init__(self, config_path: str = "config.json"):
        """
        Initialize config loader.

        Args:
            config_path: Path to config.json file (default: "config.json")
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = json.load(f)

        return config

    def get(self, *keys, default=None):
        """
        Get nested configuration value using dot notation or multiple keys.

        Args:
            *keys: Configuration keys to traverse (e.g., 'stock', 'ticker')
            default: Default value if key not found

        Returns:
            Configuration value or default if not found

        Examples:
            config.get('stock', 'ticker')
            config.get('model', 'gru', 'hidden_size')
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_stock_config(self) -> Dict[str, Any]:
        """Get stock configuration"""
        return self.config.get('stock', {})

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.get('model', {})

    def get_gru_config(self) -> Dict[str, Any]:
        """Get GRU model specific configuration"""
        return self.config.get('model', {}).get('gru', {})

    def get_sentiment_config(self) -> Dict[str, Any]:
        """Get sentiment model specific configuration"""
        return self.config.get('model', {}).get('sentiment', {})

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config.get('training', {})

    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration"""
        return self.config.get('paths', {})

    def get_discord_config(self) -> Dict[str, Any]:
        """Get Discord configuration"""
        return self.config.get('discord', {})

    def get_execution_config(self) -> Dict[str, Any]:
        """Get execution configuration"""
        return self.config.get('execution', {})

    def reload(self):
        """Reload configuration from file"""
        self.config = self._load_config()

    def save(self):
        """Save current configuration back to file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def update(self, *keys, value):
        """
        Update a configuration value.

        Args:
            *keys: Configuration keys to traverse
            value: New value to set

        Examples:
            config.update('stock', 'ticker', value='TSLA')
        """
        target = self.config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value


# Global config instance
_global_config = None

def get_config(config_path: str = "config.json") -> ConfigLoader:
    """
    Get global configuration instance (singleton pattern).

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        ConfigLoader instance
    """
    global _global_config
    if _global_config is None:
        _global_config = ConfigLoader(config_path)
    return _global_config


def reload_config():
    """Reload global configuration from file"""
    global _global_config
    if _global_config is not None:
        _global_config.reload()
