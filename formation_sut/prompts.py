"""大模型 prompt 模板：设计作战单位编成。"""
from typing import List

from schemas import Unit

SYSTEM_PROMPT_TEMPLATE = """你是一名军事编队参谋助手。作战单位库是封闭固定的，仅包含以下 {n} 种单位，禁止输出单位库之外的单位：

{unit_catalog}

本次任务的军费预算上限为 {budget_limit}（各单位的"军费消耗"乘以数量之和不能超过这个上限）。

请阅读用户给出的任务描述，结合每种单位的灵活性、机动性、隐蔽性、战斗威力等属性，判断应该选择哪些单位类型、每种配置多少数量，使编成既能满足任务需求，又不超出预算上限。

只输出严格符合以下结构的 JSON，不要输出任何解释性文字，不要使用 Markdown 代码块：

{{"units": [{{"unit_id": "单位库中的id", "count": 正整数}}]}}

要求：
1. units 数组中的 unit_id 必须是单位库中存在的 id，不得重复出现同一个 unit_id。
2. count 必须是大于 0 的整数。
3. 任务不需要的单位类型不要出现在结果中。
4. 尽量在预算上限内做出合理搭配，不要无视任务需求一味堆砌战斗威力最高的单位。
"""


def _format_catalog(units: List[Unit]) -> str:
    lines = []
    for u in units:
        attrs = "，".join(f"{k}:{v}" for k, v in u.attributes.items())
        lines.append(f"- {u.id}：{u.name}（{u.category}，{attrs}，军费消耗:{u.cost}）")
    return "\n".join(lines)


def build_messages(task_description: str, units: List[Unit], budget_limit: int) -> list:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        n=len(units), unit_catalog=_format_catalog(units), budget_limit=budget_limit
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"任务描述：{task_description}"},
    ]
