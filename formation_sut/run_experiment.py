"""批量执行 data/test_cases.jsonl 中维护的等价类/蜕变测试用例，并可选地让一个独立指定的大模型担任评判员。

用法（需先在另一个终端启动被测服务 `python app.py`）：
    python run_experiment.py
    python run_experiment.py --judge-config-name remote-scnet-deepseek
    python run_experiment.py --llm-config-names local-lmstudio-gemma,remote-scnet-deepseek --judge-config-name remote-scnet-deepseek --repeat 3

用例本身不写在这个脚本里，维护在 data/test_cases.jsonl，每行一条用例：
    id              用例编号
    group           分组标签（仅用于汇总展示）
    task_description / budget_limit  传给被测系统的参数
    call_mode       "http"（默认，正常调用 /formation/plan）或
                     "http_shuffled_catalog"（额外绕过 HTTP，直接用原始/反转的单位目录顺序各调一次，
                     用于一致性关系——公开接口无法控制目录顺序）
    ref_cases       前置/关联用例 id 数组。这些用例的输出会随本用例一起传给评判模型作为参照
    expectation     用自然语言写的预期判定要点，直接交给评判模型作为判断依据

输出：
    默认在 data/experiment_runs/<timestamp>/ 下生成 raw_results.jsonl（每条用例的完整原始记录）和 summary.csv（汇总表格）；
    可用 --raw-results-file 指定固定路径，中断后重跑会自动跳过该文件里已完成的用例。完整参数说明见文件末尾的"调用方式"注释块。
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from budget import evaluate_budget  # noqa: E402
from data_loader import load_llm_configs, load_units  # noqa: E402
from llm_client import plan_units  # noqa: E402

DEFAULT_CASES_PATH = BASE_DIR / "data" / "test_cases.jsonl"
ATTRS = ["灵活性", "机动性", "隐蔽性", "战斗威力"]
UNITS_BY_ID = {u.id: u for u in load_units()}

JUDGE_SYSTEM_PROMPT = """你是一名军事智能系统测试评判员。你会看到一次"智能编队"被测系统的输出结果，以及测试设计者给出的预期判定要点。
如果该用例引用了前置/关联用例，也会一并提供那些用例的任务描述与输出，作为判断的参照依据。

被测系统的输出中，units 是选出的作战单位编成（unit_id、count），total_cost 是总花费，over_budget 是是否超预算，
derived_metrics 是从 units 按单位属性表（灵活性/机动性/隐蔽性/战斗威力）计算出的汇总数值（overall 为全部单位汇总，
by_category 为按坦克/飞机/步兵三大类分别汇总），status_code 为 200 表示正常生成、502 表示被测系统判定输出不合法
（比如编造了单位库之外的 unit_id）。

请你判断本次输出是否符合给定的预期判定要点，只输出严格符合以下结构的 JSON，不要输出任何解释性文字之外的内容，不要使用 Markdown 代码块：

{"verdict": "pass 或 fail 或 uncertain", "reasoning": "简要说明判断依据"}

要求：
1. verdict 必须是 pass / fail / uncertain 三者之一；只有证据明显支持结论时才给 pass 或 fail，证据不足或要点本身就要求"仅供参照"时给 pass。
2. reasoning 用一两句话说清楚关键依据，尽量引用具体数值或单位名称。
"""


# ---------------------------------------------------------------------------
# 加载用例
# ---------------------------------------------------------------------------

def load_cases(path: Path) -> List[dict]:
    cases = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON: {exc}") from exc
            for field in ("id", "task_description", "budget_limit", "expectation", "ref_cases"):
                if field not in case:
                    raise ValueError(f"{path} 第 {line_no} 行缺少字段 {field}")
            case.setdefault("call_mode", "http")
            case.setdefault("group", "")
            cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# 调用被测系统 + 计算派生指标
# ---------------------------------------------------------------------------

def call_plan_http(client: httpx.Client, base_url: str, task_description: str, budget_limit: int,
                    llm_config_name: Optional[str], max_retries: int = 1) -> dict:
    payload = {"task_description": task_description, "budget_limit": budget_limit}
    if llm_config_name:
        payload["llm_config_name"] = llm_config_name

    last_error = None
    for attempt in range(max_retries + 1):
        started = time.monotonic()
        try:
            resp = client.post(f"{base_url}/formation/plan", json=payload, timeout=120)
        except httpx.HTTPError as exc:  # 被测服务偶发超时/连接异常，重试后仍失败则降级为错误结果，不中断整轮实验
            last_error = exc
            if attempt < max_retries:
                time.sleep(2 * (attempt + 1))
                continue
            elapsed_ms = round((time.monotonic() - started) * 1000)
            return {
                "status_code": 0, "elapsed_ms": elapsed_ms,
                "units": [], "total_cost": 0, "over_budget": False, "raw_llm_output": "",
                "error": f"请求被测服务失败（重试 {max_retries} 次后仍失败）: {last_error}",
            }

        elapsed_ms = round((time.monotonic() - started) * 1000)
        body = resp.json()
        if resp.status_code == 200:
            return {
                "status_code": resp.status_code, "elapsed_ms": elapsed_ms,
                "units": body["units"], "total_cost": body["total_cost"],
                "over_budget": body["over_budget"], "raw_llm_output": body["raw_llm_output"],
                "error": None,
            }
        return {
            "status_code": resp.status_code, "elapsed_ms": elapsed_ms,
            "units": [], "total_cost": 0, "over_budget": False, "raw_llm_output": "",
            "error": body.get("detail", str(body)),
        }


def call_plan_direct(task_description: str, units_list: list, budget_limit: int, config) -> dict:
    """绕过 HTTP，直接调用 plan_units——只有这样才能控制传给大模型的单位目录顺序。"""
    try:
        selections, raw = plan_units(task_description, units_list, budget_limit, config)
        unit_dicts = [u.model_dump() for u in units_list]
        budget_result = evaluate_budget(selections, unit_dicts, budget_limit)
        return {
            "status_code": 200, "elapsed_ms": None, "units": selections,
            "total_cost": budget_result["total_cost"], "over_budget": budget_result["over_budget"],
            "raw_llm_output": raw, "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status_code": 502, "elapsed_ms": None, "units": [], "total_cost": 0,
            "over_budget": False, "raw_llm_output": getattr(exc, "raw_output", ""), "error": str(exc),
        }


def compute_derived_metrics(units: List[dict]) -> dict:
    overall = {attr: 0 for attr in ATTRS}
    by_category: Dict[str, Dict[str, int]] = {}
    unit_count_by_category: Dict[str, int] = {}
    for u in units:
        unit = UNITS_BY_ID.get(u["unit_id"])
        if unit is None:
            continue
        cat = unit.category
        by_category.setdefault(cat, {attr: 0 for attr in ATTRS})
        unit_count_by_category[cat] = unit_count_by_category.get(cat, 0) + u["count"]
        for attr in ATTRS:
            value = unit.attributes.get(attr, 0) * u["count"]
            overall[attr] += value
            by_category[cat][attr] += value
    return {"overall": overall, "by_category": by_category, "unit_count_by_category": unit_count_by_category}


def execute_case(case: dict, client: httpx.Client, base_url: str, sut_config_name: Optional[str],
                  configs: dict, default_name: str, all_units: list) -> dict:
    """执行一条用例，返回会被存起来供后续用例引用、并交给评判模型的输出。"""
    if case["call_mode"] == "http":
        result = call_plan_http(client, base_url, case["task_description"], case["budget_limit"], sut_config_name)
        return {"result": result, "derived_metrics": compute_derived_metrics(result["units"])}

    if case["call_mode"] == "http_shuffled_catalog":
        config = configs[sut_config_name] if sut_config_name else configs[default_name]
        normal = call_plan_direct(case["task_description"], all_units, case["budget_limit"], config)
        shuffled = call_plan_direct(case["task_description"], list(reversed(all_units)), case["budget_limit"], config)
        return {
            "normal_order": {"result": normal, "derived_metrics": compute_derived_metrics(normal["units"])},
            "shuffled_order": {"result": shuffled, "derived_metrics": compute_derived_metrics(shuffled["units"])},
        }

    raise ValueError(f"未知 call_mode: {case['call_mode']}")


# ---------------------------------------------------------------------------
# 评判模型
# ---------------------------------------------------------------------------

def _strip_code_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
    return content


def parse_judge_response(content: str) -> dict:
    cleaned = _strip_code_fence(content)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"verdict": "uncertain", "reasoning": f"评判模型输出无法解析为 JSON: {content}"}
    verdict = data.get("verdict") if isinstance(data, dict) else None
    if verdict not in ("pass", "fail", "uncertain"):
        return {"verdict": "uncertain", "reasoning": f"评判模型输出 verdict 字段非法: {data}"}
    return {"verdict": verdict, "reasoning": data.get("reasoning", "")}


def build_judge_user_message(case: dict, case_output: dict, ref_entries: List[dict]) -> str:
    payload = {
        "本用例": {
            "id": case["id"],
            "task_description": case["task_description"],
            "budget_limit": case["budget_limit"],
            "output": case_output,
        },
        "预期判定要点": case["expectation"],
        "前置或关联用例": ref_entries,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def judge_case(judge_client: OpenAI, judge_model: str, case: dict, case_output: dict,
               ref_entries: List[dict], max_retries: int = 2) -> dict:
    user_message = build_judge_user_message(case, case_output, ref_entries)
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = judge_client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            verdict = parse_judge_response(content)
            verdict["raw_judge_output"] = content
            return verdict
        except Exception as exc:  # noqa: BLE001  评判服务偶发网关/限流类瞬时错误，重试后仍失败则降级为 uncertain，不中断整轮实验
            last_error = exc
            if attempt < max_retries:
                time.sleep(2 * (attempt + 1))
    return {
        "verdict": "uncertain",
        "reasoning": f"评判模型调用失败（重试 {max_retries} 次后仍失败）: {last_error}",
        "raw_judge_output": "",
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_one_pass(cases: List[dict], client: httpx.Client, base_url: str, sut_config_name: Optional[str],
                  configs: dict, default_name: str, all_units: list,
                  judge_client: Optional[OpenAI], judge_model: Optional[str],
                  repeat_index: int, sut_config_label: str, judge_config_label: str,
                  raw_file, done_keys: set, existing_by_key: Dict[tuple, dict]) -> List[dict]:
    case_outputs: Dict[int, dict] = {}
    rows: List[dict] = []

    for case in cases:
        key = (sut_config_label, repeat_index, case["id"])
        if key in done_keys:
            case_outputs[case["id"]] = existing_by_key[key]["output"]
            print(f"  用例{case['id']} [{case['group']}]: 跳过（已在 {raw_file.name} 中找到结果，继续上次未完成的部分）")
            continue

        missing = [ref_id for ref_id in case["ref_cases"] if ref_id not in case_outputs]
        if missing:
            raise RuntimeError(
                f"用例 {case['id']} 引用的前置用例 {missing} 还没有结果，"
                f"请检查 data/test_cases.jsonl 中的用例顺序（被引用的用例必须排在前面）。"
            )

        output = execute_case(case, client, base_url, sut_config_name, configs, default_name, all_units)
        case_outputs[case["id"]] = output

        judge_result = None
        if judge_client is not None:
            ref_entries = [
                {"id": ref_id, "task_description": next(c["task_description"] for c in cases if c["id"] == ref_id),
                 "output": case_outputs[ref_id]}
                for ref_id in case["ref_cases"]
            ]
            judge_result = judge_case(judge_client, judge_model, case, output, ref_entries)

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sut_config_name": sut_config_label,
            "judge_config_name": judge_config_label,
            "repeat_index": repeat_index,
            "case_id": case["id"],
            "group": case["group"],
            "task_description": case["task_description"],
            "budget_limit": case["budget_limit"],
            "ref_cases": case["ref_cases"],
            "expectation": case["expectation"],
            "output": output,
            "verdict": judge_result["verdict"] if judge_result else "not_judged",
            "reasoning": judge_result["reasoning"] if judge_result else "",
            "raw_judge_output": judge_result["raw_judge_output"] if judge_result else "",
        }
        rows.append(row)
        raw_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        raw_file.flush()

        mark = {"pass": "PASS", "fail": "FAIL", "uncertain": "不确定", "not_judged": "未评判"}[row["verdict"]]
        print(f"  用例{case['id']} [{case['group']}]: {mark}" + (f" —— {row['reasoning']}" if row["reasoning"] else ""))

    return rows


def flatten_units(output: dict) -> str:
    if "result" in output:
        return json.dumps(output["result"]["units"], ensure_ascii=False)
    return json.dumps({k: v["result"]["units"] for k, v in output.items()}, ensure_ascii=False)


def load_existing_rows(raw_path: Path) -> List[dict]:
    if not raw_path.exists():
        return []
    rows = []
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="批量执行 data/test_cases.jsonl 中维护的测试用例")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases-file", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--llm-config-names", default="", help="被测系统使用的 llm_config_name，逗号分隔，留空则只用服务端默认配置跑一次")
    parser.add_argument("--judge-config-name", default="", help="评判模型使用的 llm_config_name（来自 config/llm_config.json）。留空则不做 LLM 评判，只采集原始结果")
    parser.add_argument("--repeat", type=int, default=1, help="每个配置重复跑几遍（用于观察大模型输出的稳定性）")
    parser.add_argument("--out-dir", default=str(BASE_DIR / "data" / "experiment_runs"),
                         help="未指定 --raw-results-file 时，自动在此目录下新建一个带时间戳的子目录存放结果")
    parser.add_argument("--raw-results-file", default="",
                         help="直接指定 raw_results.jsonl 的输出路径（summary.csv 会写在同一目录下）。"
                              "若该文件已存在，会先加载里面已完成的用例记录并跳过，只补跑剩余的，"
                              "用于中断后继续，避免从头重跑。不指定则按 --out-dir 规则新建一个带时间戳的目录。")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases_file))
    sut_config_names = [n.strip() for n in args.llm_config_names.split(",") if n.strip()] or [None]

    configs, default_name = load_llm_configs()
    all_units = load_units()

    judge_client = None
    judge_model = None
    if args.judge_config_name:
        if args.judge_config_name not in configs:
            print(f"未知的 judge-config-name: {args.judge_config_name}，可选: {list(configs)}")
            sys.exit(1)
        judge_config = configs[args.judge_config_name]
        judge_client = OpenAI(base_url=judge_config.base_url, api_key=judge_config.api_key,
                               timeout=judge_config.timeout_seconds)
        judge_model = judge_config.model

    if args.raw_results_file:
        raw_path = Path(args.raw_results_file)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.out_dir) / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / "raw_results.jsonl"
    csv_path = raw_path.parent / "summary.csv"

    # 若指定的 raw_results.jsonl 已存在（中断重跑场景），先加载已完成的用例记录，
    # 之后只补跑缺的部分；run_one_pass 内部按 (sut_config_label, repeat_index, case_id) 跳过已完成用例。
    existing_rows = load_existing_rows(raw_path)
    done_keys = {(r["sut_config_name"], r["repeat_index"], r["case_id"]) for r in existing_rows}
    existing_by_key = {(r["sut_config_name"], r["repeat_index"], r["case_id"]): r for r in existing_rows}
    all_rows: List[dict] = list(existing_rows)
    if existing_rows:
        print(f"检测到 {raw_path} 中已有 {len(existing_rows)} 条记录，将跳过已完成用例，从中断处继续。")

    # raw_results.jsonl 边跑边写（每条用例落盘一行，追加模式），这样某个配置/重复轮中途因远端服务瞬时报错等
    # 原因中断时，已经跑完、已经付费调用过评判模型的用例结果不会丢失，重跑时也能被上面的加载逻辑识别并跳过。
    with httpx.Client() as client, raw_path.open("a", encoding="utf-8") as raw_file:
        try:
            client.get(f"{args.base_url}/units", timeout=5)
        except httpx.HTTPError as exc:
            print(f"无法连接到 {args.base_url}，请先启动被测服务（python app.py）。错误: {exc}")
            sys.exit(1)

        judge_label = args.judge_config_name or "(none)"
        for sut_config_name in sut_config_names:
            for repeat_index in range(args.repeat):
                label = sut_config_name or "(default)"
                print(f"=== 被测配置: {label} | 评判配置: {args.judge_config_name or '(不评判)'} | 第 {repeat_index + 1}/{args.repeat} 轮 ===")
                rows = run_one_pass(cases, client, args.base_url, sut_config_name, configs, default_name,
                                     all_units, judge_client, judge_model, repeat_index, label, judge_label,
                                     raw_file, done_keys, existing_by_key)
                all_rows.extend(rows)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sut_config_name", "judge_config_name", "repeat_index", "case_id",
                                                "group", "task_description", "budget_limit", "units", "verdict",
                                                "reasoning"])
        writer.writeheader()
        for row in all_rows:
            writer.writerow({
                "sut_config_name": row["sut_config_name"], "judge_config_name": row["judge_config_name"],
                "repeat_index": row["repeat_index"], "case_id": row["case_id"], "group": row["group"],
                "task_description": row["task_description"], "budget_limit": row["budget_limit"],
                "units": flatten_units(row["output"]), "verdict": row["verdict"], "reasoning": row["reasoning"],
            })

    fail_count = sum(1 for r in all_rows if r["verdict"] == "fail")
    uncertain_count = sum(1 for r in all_rows if r["verdict"] == "uncertain")
    not_judged_count = sum(1 for r in all_rows if r["verdict"] == "not_judged")
    print(f"\n完成。fail: {fail_count}；uncertain: {uncertain_count}；未评判: {not_judged_count}。")
    print(f"原始记录: {raw_path}")
    print(f"汇总表格: {csv_path}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 调用方式
# ---------------------------------------------------------------------------
# 前提：先在另一个终端启动被测服务 `python app.py`（默认监听 127.0.0.1:8000）。
#
# 最简单的跑法（只用服务端默认 LLM 配置跑一遍，不做自动评判）：
#     python run_experiment.py
#
# 参数说明：
#   --base-url            被测服务地址，默认 http://127.0.0.1:8000
#   --cases-file          测试用例文件路径，默认 data/test_cases.jsonl
#   --llm-config-names    被测系统使用的 llm_config_name，逗号分隔多个可对比不同模型；
#                         留空则只用服务端默认配置跑一次
#                         例：--llm-config-names local-lmstudio-gemma,remote-scnet-minimax
#   --judge-config-name   评判模型使用的 llm_config_name（取自 config/llm_config.json）；
#                         留空则不做 LLM 自动评判，只采集原始结果
#                         例：--judge-config-name remote-scnet-minimax
#   --repeat              每个配置重复跑几遍，用于观察大模型输出的稳定性，默认 1
#   --out-dir             未指定 --raw-results-file 时，自动在此目录下新建带时间戳的子目录存放结果，
#                         默认 data/experiment_runs
#   --raw-results-file    直接指定 raw_results.jsonl 的输出路径（summary.csv 写在同一目录下）。
#                         若该文件已存在，会先加载里面已完成的用例记录并跳过，只补跑剩余的，
#                         用于中断后继续，避免从头重跑；不指定则按 --out-dir 规则新建带时间戳的目录
#                         例：--raw-results-file data/experiment_runs/manual_run/raw_results.jsonl
#
# 完整示例（两个被测配置各跑 3 遍，并用指定模型自动评判）：
#     python run_experiment.py --llm-config-names local-lmstudio-gemma --judge-config-name remote-scnet-minimax --repeat 3
#
# 中断后继续跑（指定固定输出文件，脚本异常退出或被手动终止后，原样加上同一个 --raw-results-file 重新执行即可，
# 已完成的用例会被跳过，只补跑剩余的）：
#     python run_experiment.py --llm-config-names local-lmstudio-gemma --judge-config-name remote-scnet-minimax --repeat 1  --raw-results-file data/experiment_runs/manual_run/raw_results.jsonl
#
# 输出：
#     <输出目录>/raw_results.jsonl  每条用例的完整原始记录（含评判结果），边跑边追加写入
#     <输出目录>/summary.csv         每条用例一行的判定结果汇总（每次运行结束后基于当前 raw_results.jsonl 全量重写）
