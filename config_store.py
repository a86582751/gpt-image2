import json
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("app_config.json")


def load_config(default_config):
    if not CONFIG_PATH.exists():
        return default_config.copy()

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            saved_config = json.load(config_file)
    except (OSError, json.JSONDecodeError):
        return default_config.copy()

    config = default_config.copy()
    if isinstance(saved_config, dict):
        config.update({key: value for key, value in saved_config.items() if key in config})
    return config


def save_config(config):
    temp_path = CONFIG_PATH.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=2)
    temp_path.replace(CONFIG_PATH)


def update_config(default_config, updates):
    config = load_config(default_config)
    config.update(updates)
    save_config(config)
