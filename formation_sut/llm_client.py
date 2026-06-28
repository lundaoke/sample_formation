"""封装大模型调用：设计作战单位编成。"""
import json
from typing import List, Tuple

from openai import OpenAI

from prompts import build_messages
from schemas import LLMConfig, Unit


class LLMOutputError(Exception):
    """大模型输出无法解析为合法、自洽的单位编成方案。"""

    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


def _strip_code_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
    return content


def parse_plan_response(content: str, units: List[Unit]) -> list:
    """解析并校验大模型返回的单位编成 JSON，返回合并了单位库信息的方案列表。"""
    cleaned = _strip_code_fence(content)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"大模型输出不是合法 JSON: {exc}\n原始输出: {content}", raw_output=content) from exc

    selections = data.get("units") if isinstance(data, dict) else None
    if not isinstance(selections, list) or not selections:
        raise LLMOutputError(f"大模型输出缺少非空的 units 数组: {content}", raw_output=content)

    catalog = {u.id: u for u in units}
    seen_ids = set()
    result = []
    for item in selections:
        unit_id = item.get("unit_id") if isinstance(item, dict) else None
        count = item.get("count") if isinstance(item, dict) else None

        if unit_id not in catalog:
            raise LLMOutputError(f"大模型输出了单位库之外的 unit_id: {unit_id}", raw_output=content)
        if unit_id in seen_ids:
            raise LLMOutputError(f"大模型重复输出了同一单位: {unit_id}", raw_output=content)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise LLMOutputError(f"单位 {unit_id} 的 count 不是正整数: {count}", raw_output=content)

        seen_ids.add(unit_id)
        unit = catalog[unit_id]
        result.append({"unit_id": unit.id, "unit_name": unit.name, "count": count})
    return result


def plan_units(task_description: str, units: List[Unit], budget_limit: int, config: LLMConfig) -> Tuple[list, str]:
    """调用大模型设计作战单位编成，返回 (解析后的单位列表, 原始响应文本)。"""
    client = OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=config.timeout_seconds)
    messages = build_messages(task_description, units, budget_limit)
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
    )
    content = response.choices[0].message.content or ""
    return parse_plan_response(content, units), content
