"""加载 units.json / llm_config.json 到内存。"""
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

from schemas import LLMConfig, Unit

BASE_DIR = Path(__file__).resolve().parent
UNITS_PATH = BASE_DIR / "data" / "units.json"
LLM_CONFIG_PATH = BASE_DIR / "config" / "llm_config.json"


@lru_cache(maxsize=1)
def load_units() -> List[Unit]:
    data = json.loads(UNITS_PATH.read_text(encoding="utf-8"))
    return [Unit.model_validate(item) for item in data]


@lru_cache(maxsize=1)
def load_llm_configs() -> Tuple[Dict[str, LLMConfig], str]:
    """返回 (配置名 -> LLMConfig 的字典, 默认配置名)。"""
    data = json.loads(LLM_CONFIG_PATH.read_text(encoding="utf-8"))
    configs = {name: LLMConfig.model_validate(value) for name, value in data["models"].items()}
    return configs, data["default"]
