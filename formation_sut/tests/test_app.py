import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

import app as app_module
from llm_client import LLMOutputError

client = TestClient(app_module.app)


def test_get_units_returns_seven_units():
    response = client.get("/units")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 7
    assert {u["id"] for u in body} == {
        "heavy_tank", "light_tank", "recon_vehicle",
        "attack_helicopter", "stealth_recon_aircraft",
        "infantry_squad", "special_forces_team",
    }


def test_get_config_lists_models_with_masked_api_key():
    response = client.get("/config")
    assert response.status_code == 200
    body = response.json()
    assert body["default"] in body["models"]
    for cfg in body["models"].values():
        assert "api_key_masked" in cfg
        assert "base_url" in cfg
        assert "model" in cfg


def test_formation_plan_returns_structured_result(monkeypatch, tmp_path):
    logged = []
    monkeypatch.setattr(
        app_module,
        "append_record",
        lambda **kwargs: logged.append(kwargs),
    )

    def fake_plan_units(task_description, units, budget_limit, config):
        return (
            [
                {"unit_id": "heavy_tank", "unit_name": "主战坦克", "count": 3},
                {"unit_id": "infantry_squad", "unit_name": "普通步兵班", "count": 3},
            ],
            '{"units": [{"unit_id": "heavy_tank", "count": 3}, {"unit_id": "infantry_squad", "count": 3}]}',
        )

    monkeypatch.setattr(app_module, "plan_units", fake_plan_units)

    response = client.post(
        "/formation/plan",
        json={"task_description": "压制敌方装甲并占领据点", "budget_limit": 300},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["task_description"] == "压制敌方装甲并占领据点"
    assert body["budget_limit"] == 300
    assert [u["unit_id"] for u in body["units"]] == ["heavy_tank", "infantry_squad"]
    assert body["total_cost"] == 3 * 90 + 3 * 15
    assert body["over_budget"] is True
    assert "heavy_tank" in body["raw_llm_output"]
    assert len(logged) == 1
    assert logged[0]["task_description"] == "压制敌方装甲并占领据点"
    assert logged[0]["over_budget"] is True


def test_formation_plan_uses_default_budget_when_omitted(monkeypatch):
    monkeypatch.setattr(app_module, "append_record", lambda **kwargs: None)

    def fake_plan_units(task_description, units, budget_limit, config):
        assert budget_limit == 300
        return ([{"unit_id": "infantry_squad", "unit_name": "普通步兵班", "count": 1}], "{}")

    monkeypatch.setattr(app_module, "plan_units", fake_plan_units)

    response = client.post("/formation/plan", json={"task_description": "任意任务描述"})
    assert response.status_code == 200
    assert response.json()["budget_limit"] == 300


def test_formation_plan_rejects_unknown_llm_config_name():
    response = client.post(
        "/formation/plan",
        json={"task_description": "任意任务描述", "llm_config_name": "does-not-exist"},
    )
    assert response.status_code == 400


def test_formation_plan_returns_502_on_llm_output_error(monkeypatch):
    monkeypatch.setattr(app_module, "append_record", lambda **kwargs: None)

    def fake_plan_units(task_description, units, budget_limit, config):
        raise LLMOutputError("大模型输出不合法")

    monkeypatch.setattr(app_module, "plan_units", fake_plan_units)

    response = client.post("/formation/plan", json={"task_description": "任意任务描述"})
    assert response.status_code == 502


def test_formation_plan_logs_failed_attempt_on_llm_output_error(monkeypatch):
    logged = []
    monkeypatch.setattr(
        app_module,
        "append_record",
        lambda **kwargs: logged.append(kwargs),
    )

    def fake_plan_units(task_description, units, budget_limit, config):
        raise LLMOutputError("大模型输出了单位库之外的 unit_id: special_forces_squad", raw_output='{"units": [{"unit_id": "special_forces_squad", "count": 1}]}')

    monkeypatch.setattr(app_module, "plan_units", fake_plan_units)

    response = client.post("/formation/plan", json={"task_description": "任意任务描述", "budget_limit": 300})
    assert response.status_code == 502
    assert len(logged) == 1
    assert logged[0]["task_description"] == "任意任务描述"
    assert logged[0]["budget_limit"] == 300
    assert logged[0]["units"] == []
    assert logged[0]["total_cost"] == 0
    assert logged[0]["over_budget"] is False
    assert "special_forces_squad" in logged[0]["raw_llm_output"]
    assert "special_forces_squad" in logged[0]["error"]


def test_formation_plan_logs_failed_attempt_on_generic_exception(monkeypatch):
    logged = []
    monkeypatch.setattr(
        app_module,
        "append_record",
        lambda **kwargs: logged.append(kwargs),
    )

    def fake_plan_units(task_description, units, budget_limit, config):
        raise RuntimeError("连接超时")

    monkeypatch.setattr(app_module, "plan_units", fake_plan_units)

    response = client.post("/formation/plan", json={"task_description": "任意任务描述"})
    assert response.status_code == 502
    assert len(logged) == 1
    assert logged[0]["raw_llm_output"] == ""
    assert "连接超时" in logged[0]["error"]


def test_get_history_returns_records_from_read_records(monkeypatch):
    fake_records = [
        {
            "timestamp": "2026-06-17T00:00:00+00:00",
            "task_description": "任务A",
            "budget_limit": 300,
            "units": [{"unit_id": "infantry_squad", "unit_name": "普通步兵班", "count": 2}],
            "total_cost": 30,
            "over_budget": False,
            "raw_llm_output": "{}",
            "error": None,
        }
    ]
    monkeypatch.setattr(app_module, "read_records", lambda: fake_records)

    response = client.get("/history")
    assert response.status_code == 200
    body = response.json()
    assert body == fake_records


def test_get_history_returns_empty_list_when_no_records(monkeypatch):
    monkeypatch.setattr(app_module, "read_records", lambda: [])

    response = client.get("/history")
    assert response.status_code == 200
    assert response.json() == []


def test_history_page_is_served():
    response = client.get("/history-page")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
