import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from llm_client import LLMOutputError, parse_plan_response
from schemas import Unit

UNITS = [
    Unit(id="heavy_tank", name="主战坦克", category="坦克", attributes={"灵活性": 3, "机动性": 3, "隐蔽性": 1, "战斗威力": 9}, cost=90),
    Unit(id="infantry_squad", name="普通步兵班", category="步兵", attributes={"灵活性": 6, "机动性": 4, "隐蔽性": 5, "战斗威力": 4}, cost=15),
]


def test_parses_valid_json():
    content = '{"units": [{"unit_id": "infantry_squad", "count": 3}]}'
    result = parse_plan_response(content, UNITS)
    assert result == [{"unit_id": "infantry_squad", "unit_name": "普通步兵班", "count": 3}]


def test_strips_markdown_code_fence():
    content = '```json\n{"units": [{"unit_id": "infantry_squad", "count": 3}]}\n```'
    result = parse_plan_response(content, UNITS)
    assert result[0]["unit_id"] == "infantry_squad"


def test_rejects_invalid_json():
    with pytest.raises(LLMOutputError):
        parse_plan_response("not json at all", UNITS)


def test_rejects_missing_units_key():
    with pytest.raises(LLMOutputError):
        parse_plan_response("{}", UNITS)


def test_rejects_empty_units_array():
    with pytest.raises(LLMOutputError):
        parse_plan_response('{"units": []}', UNITS)


def test_rejects_unknown_unit_id():
    content = '{"units": [{"unit_id": "fighter_jet", "count": 1}]}'
    with pytest.raises(LLMOutputError):
        parse_plan_response(content, UNITS)


def test_rejects_duplicate_unit_id():
    content = (
        '{"units": ['
        '{"unit_id": "infantry_squad", "count": 1},'
        '{"unit_id": "infantry_squad", "count": 2}'
        "]}"
    )
    with pytest.raises(LLMOutputError):
        parse_plan_response(content, UNITS)


def test_rejects_non_positive_count():
    content = '{"units": [{"unit_id": "infantry_squad", "count": 0}]}'
    with pytest.raises(LLMOutputError):
        parse_plan_response(content, UNITS)


def test_error_carries_raw_output_for_unknown_unit_id():
    content = '{"units": [{"unit_id": "fighter_jet", "count": 1}]}'
    with pytest.raises(LLMOutputError) as exc_info:
        parse_plan_response(content, UNITS)
    assert exc_info.value.raw_output == content


def test_error_carries_raw_output_for_invalid_json():
    content = "not json at all"
    with pytest.raises(LLMOutputError) as exc_info:
        parse_plan_response(content, UNITS)
    assert exc_info.value.raw_output == content
