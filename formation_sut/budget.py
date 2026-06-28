"""确定性预算核算（第二步：不调用大模型）。"""
from typing import Mapping, Sequence


def compute_total_cost(selections: Sequence[Mapping], catalog: Sequence[Mapping]) -> int:
    cost_by_id = {u["id"]: u["cost"] for u in catalog}
    return sum(cost_by_id[s["unit_id"]] * s["count"] for s in selections)


def evaluate_budget(selections: Sequence[Mapping], catalog: Sequence[Mapping], budget_limit: int) -> dict:
    total_cost = compute_total_cost(selections, catalog)
    return {"total_cost": total_cost, "over_budget": total_cost > budget_limit}
