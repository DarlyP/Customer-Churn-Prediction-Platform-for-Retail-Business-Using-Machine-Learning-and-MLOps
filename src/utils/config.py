from pathlib import Path
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load project configuration from YAML file.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config