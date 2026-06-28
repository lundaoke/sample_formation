"""Pydantic 数据模型：请求体、响应体、配置与数据实体。"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

AttributeVector = Dict[str, int]


class Unit(BaseModel):
    id: str
    name: str
    category: str
    attributes: AttributeVector
    cost: int


class LLMConfig(BaseModel):
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.3
    timeout_seconds: int = 60


class FormationRequest(BaseModel):
    task_description: str = Field(min_length=1)
    budget_limit: int = Field(default=300, gt=0)
    llm_config_name: Optional[str] = None


class UnitSelection(BaseModel):
    unit_id: str
    unit_name: str
    count: int


class FormationResponse(BaseModel):
    task_description: str
    budget_limit: int
    units: List[UnitSelection]
    total_cost: int
    over_budget: bool
    raw_llm_output: str


class HistoryRecord(BaseModel):
    timestamp: str
    task_description: str
    budget_limit: int
    units: List[UnitSelection]
    total_cost: int
    over_budget: bool
    raw_llm_output: str
    error: Optional[str] = None
