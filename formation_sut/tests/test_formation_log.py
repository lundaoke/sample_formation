import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formation_log import append_record, read_records


def test_append_record_writes_one_jsonl_line(tmp_path):
    log_path = tmp_path / "formation_log.jsonl"
    units = [{"unit_id": "infantry_squad", "unit_name": "普通步兵班", "count": 3}]

    append_record(
        task_description="组织一次步兵突击",
        budget_limit=300,
        units=units,
        total_cost=45,
        over_budget=False,
        raw_llm_output='{"units": [{"unit_id": "infantry_squad", "count": 3}]}',
        log_path=log_path,
    )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["task_description"] == "组织一次步兵突击"
    assert record["budget_limit"] == 300
    assert record["units"] == units
    assert record["total_cost"] == 45
    assert record["over_budget"] is False
    assert "infantry_squad" in record["raw_llm_output"]
    assert "timestamp" in record


def test_append_record_includes_parseable_iso_timestamp(tmp_path):
    log_path = tmp_path / "formation_log.jsonl"
    append_record(
        task_description="任务A",
        budget_limit=300,
        units=[],
        total_cost=0,
        over_budget=False,
        raw_llm_output="{}",
        log_path=log_path,
    )
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    datetime.fromisoformat(record["timestamp"])


def test_append_record_appends_without_overwriting(tmp_path):
    log_path = tmp_path / "formation_log.jsonl"
    common = dict(budget_limit=300, units=[], total_cost=0, over_budget=False, raw_llm_output="{}")
    append_record(task_description="任务A", log_path=log_path, **common)
    append_record(task_description="任务B", log_path=log_path, **common)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["task_description"] == "任务A"
    assert json.loads(lines[1])["task_description"] == "任务B"


def test_append_record_creates_parent_directory(tmp_path):
    log_path = tmp_path / "nested" / "dir" / "formation_log.jsonl"
    append_record(
        task_description="任务A",
        budget_limit=300,
        units=[],
        total_cost=0,
        over_budget=False,
        raw_llm_output="{}",
        log_path=log_path,
    )
    assert log_path.exists()


def test_read_records_returns_empty_list_when_file_missing(tmp_path):
    log_path = tmp_path / "does_not_exist.jsonl"
    assert read_records(log_path=log_path) == []


def test_read_records_returns_most_recent_first(tmp_path):
    log_path = tmp_path / "formation_log.jsonl"
    common = dict(budget_limit=300, units=[], total_cost=0, over_budget=False, raw_llm_output="{}")
    append_record(task_description="任务A", log_path=log_path, **common)
    append_record(task_description="任务B", log_path=log_path, **common)
    append_record(task_description="任务C", log_path=log_path, **common)

    records = read_records(log_path=log_path)

    assert [r["task_description"] for r in records] == ["任务C", "任务B", "任务A"]


def test_append_record_writes_error_field_when_provided(tmp_path):
    log_path = tmp_path / "formation_log.jsonl"
    append_record(
        task_description="任务A",
        budget_limit=300,
        units=[],
        total_cost=0,
        over_budget=False,
        raw_llm_output="not json at all",
        error="大模型输出不是合法 JSON",
        log_path=log_path,
    )
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["error"] == "大模型输出不是合法 JSON"


def test_append_record_defaults_error_field_to_none(tmp_path):
    log_path = tmp_path / "formation_log.jsonl"
    append_record(
        task_description="任务A",
        budget_limit=300,
        units=[],
        total_cost=0,
        over_budget=False,
        raw_llm_output="{}",
        log_path=log_path,
    )
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["error"] is None


def test_read_records_skips_blank_lines(tmp_path):
    log_path = tmp_path / "formation_log.jsonl"
    append_record(task_description="任务A", log_path=log_path, budget_limit=300, units=[], total_cost=0, over_budget=False, raw_llm_output="{}")
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n")

    records = read_records(log_path=log_path)

    assert len(records) == 1
