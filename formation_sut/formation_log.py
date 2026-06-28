"""记录每次编队任务的输入与输出，便于实验复盘与测试结果回溯。"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = BASE_DIR / "data" / "formation_log.jsonl"


def append_record(
    task_description: str,
    budget_limit: int,
    units: list,
    total_cost: int,
    over_budget: bool,
    raw_llm_output: str,
    error: Optional[str] = None,
    log_path: Path = DEFAULT_LOG_PATH,
) -> dict:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_description": task_description,
        "budget_limit": budget_limit,
        "units": units,
        "total_cost": total_cost,
        "over_budget": over_budget,
        "raw_llm_output": raw_llm_output,
        "error": error,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_records(log_path: Path = DEFAULT_LOG_PATH) -> list:
    """读取历史记录，最近一次的排在最前面。"""
    if not log_path.exists():
        return []
    records = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.reverse()
    return records
