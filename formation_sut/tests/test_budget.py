import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from budget import compute_total_cost, evaluate_budget

CATALOG = [
    {"id": "heavy_tank", "cost": 90},
    {"id": "infantry_squad", "cost": 15},
]


def test_compute_total_cost_sums_selected_units():
    selections = [{"unit_id": "heavy_tank", "count": 2}, {"unit_id": "infantry_squad", "count": 3}]
    assert compute_total_cost(selections, CATALOG) == 2 * 90 + 3 * 15


def test_compute_total_cost_empty_selection_is_zero():
    assert compute_total_cost([], CATALOG) == 0


def test_evaluate_budget_marks_over_budget_true_when_exceeds():
    selections = [{"unit_id": "heavy_tank", "count": 4}]
    result = evaluate_budget(selections, CATALOG, budget_limit=300)
    assert result == {"total_cost": 360, "over_budget": True}


def test_evaluate_budget_marks_over_budget_false_when_within_limit():
    selections = [{"unit_id": "heavy_tank", "count": 2}]
    result = evaluate_budget(selections, CATALOG, budget_limit=300)
    assert result == {"total_cost": 180, "over_budget": False}


def test_evaluate_budget_exactly_at_limit_is_not_over_budget():
    selections = [{"unit_id": "heavy_tank", "count": 1}, {"unit_id": "infantry_squad", "count": 14}]
    result = evaluate_budget(selections, CATALOG, budget_limit=300)
    assert result == {"total_cost": 300, "over_budget": False}
