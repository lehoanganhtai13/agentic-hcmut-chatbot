"""
Configuration management for the web crawler.
"""

import os
import yaml


class Config:
    """
    Configuration manager for the web crawler.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize configuration from a YAML file or defaults.
        
        Args:
            config_file (str, optional): Path to YAML config file.
        """
        self.config = {
            "crawler": {
                "headless": True,
                "max_depth": 3,
                "download_images": False
            },
            "output": {
                "format": "txt",  # "txt" or "pdf"
                "directory": "data"
            },
            "logging": {
                "level": "INFO",
                "file_logging": False,
                "log_dir": "logs"
            }
        }
        
        if config_file and os.path.exists(config_file):
            self.load_from_yaml(config_file)
    
    def load_from_yaml(self, config_file):
        """
        Load configuration from a YAML file.
        
        Args:
            config_file (str): Path to YAML config file.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with open(config_file, "r") as f:
                yaml_config = yaml.safe_load(f)
            
            # Update config with values from YAML
            if yaml_config:
                for section in self.config:
                    if section in yaml_config:
                        self.config[section].update(yaml_config[section])
            return True
        except Exception as e:
            print(f"Error loading config file: {e}")
            return False
    
    def get(self, section, key=None):
        """
        Get configuration value.
        
        Args:
            section (str): Configuration section.
            key (str, optional): Configuration key.
            
        Returns:
            Any: Configuration value or section dict if key is None.
        """
        if section not in self.config:
            return None
            
        if key is None:
            return self.config[section]
            
        return self.config[section].get(key)
    