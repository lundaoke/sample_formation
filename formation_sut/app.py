"""FastAPI 入口：定义 HTTP 路由，串联大模型作战单位编成决策与确定性预算核算。"""
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from budget import evaluate_budget
from data_loader import load_llm_configs, load_units
from formation_log import append_record, read_records
from llm_client import LLMOutputError, plan_units
from schemas import FormationRequest, FormationResponse, HistoryRecord, Unit

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="智能编队 SUT")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/history-page")
def history_page():
    return FileResponse(BASE_DIR / "static" / "history.html")


@app.get("/history", response_model=List[HistoryRecord])
def get_history():
    return read_records()


@app.get("/units", response_model=List[Unit])
def get_units():
    return load_units()


def _mask_api_key(api_key: str) -> str:
    if len(api_key) <= 4:
        return "*" * len(api_key)
    return api_key[:2] + "*" * (len(api_key) - 4) + api_key[-2:]


@app.get("/config")
def get_config():
    """展示所有可选的 LLM 连接配置，api_key 做掩码处理避免泄露。"""
    configs, default_name = load_llm_configs()
    return {
        "default": default_name,
        "models": {
            name: {
                "base_url": cfg.base_url,
                "model": cfg.model,
                "temperature": cfg.temperature,
                "timeout_seconds": cfg.timeout_seconds,
                "api_key_masked": _mask_api_key(cfg.api_key),
            }
            for name, cfg in configs.items()
        },
    }


@app.post("/formation/plan", response_model=FormationResponse)
def formation_plan(request: FormationRequest):
    units = load_units()
    configs, default_name = load_llm_configs()
    config_name = request.llm_config_name or default_name
    if config_name not in configs:
        raise HTTPException(status_code=400, detail=f"未知的 llm_config_name: {config_name}")
    config = configs[config_name]

    try:
        selections, raw_output = plan_units(request.task_description, units, request.budget_limit, config)
    except LLMOutputError as exc:
        append_record(
            task_description=request.task_description,
            budget_limit=request.budget_limit,
            units=[],
            total_cost=0,
            over_budget=False,
            raw_llm_output=getattr(exc, "raw_output", ""),
            error=str(exc),
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # openai SDK 的网络/鉴权异常类型不固定，统一在此边界转换为 502
        detail = f"调用大模型失败: {exc}"
        append_record(
            task_description=request.task_description,
            budget_limit=request.budget_limit,
            units=[],
            total_cost=0,
            over_budget=False,
            raw_llm_output="",
            error=detail,
        )
        raise HTTPException(status_code=502, detail=detail) from exc

    unit_dicts = [u.model_dump() for u in units]
    budget_result = evaluate_budget(selections, unit_dicts, request.budget_limit)

    append_record(
        task_description=request.task_description,
        budget_limit=request.budget_limit,
        units=selections,
        total_cost=budget_result["total_cost"],
        over_budget=budget_result["over_budget"],
        raw_llm_output=raw_output,
    )

    return FormationResponse(
        task_description=request.task_description,
        budget_limit=request.budget_limit,
        units=selections,
        total_cost=budget_result["total_cost"],
        over_budget=budget_result["over_budget"],
        raw_llm_output=raw_output,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
