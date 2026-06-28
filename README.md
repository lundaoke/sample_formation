# 智能编队实验 (formation_sut)

基于大模型的"智能编队"决策被测系统（SUT），用于智能测试技术相关实验。系统接收任务描述与预算限制，调用大模型从作战单位目录中选择编成方案，并对结果进行确定性的预算核算。

## 项目结构

```
formation_sut/
├── app.py              # FastAPI 入口，定义 HTTP 路由
├── llm_client.py       # 调用大模型生成编成方案
├── budget.py           # 确定性预算核算
├── data_loader.py       # 加载 units.json / llm_config.json
├── formation_log.py     # 历史记录读写
├── schemas.py           # Pydantic 数据模型
├── run_experiment.py    # 批量执行测试用例（等价类/蜕变测试）
├── config/
│   └── llm_config.example.json  # LLM 连接配置示例（真实配置不提交）
├── data/                # units.json / test_cases.jsonl 等运行数据（不提交）
├── static/              # 前端页面（编成页 / 历史页）
└── tests/                # pytest 单元测试
```

## 环境准备

```bash
cd formation_sut
pip install -r requirements.txt
```

复制 `config/llm_config.example.json` 为 `config/llm_config.json`，并填入真实的 `api_key` / `base_url` 等信息（该文件已加入 `.gitignore`，不会被提交）。

## 启动服务

```bash
cd formation_sut
python app.py
```

默认监听 `http://127.0.0.1:8000`，访问 `/` 查看编成页面，`/history-page` 查看历史记录页面。

主要接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/units` | 获取作战单位目录 |
| GET | `/config` | 获取可选的 LLM 连接配置（api_key 已掩码） |
| GET | `/history` | 获取历史编成记录 |
| POST | `/formation/plan` | 提交任务描述与预算限制，生成编成方案 |

## 批量实验 / 测试用例执行

测试用例维护在 `data/test_cases.jsonl`，每行一条用例（包含分组、任务描述、预算限制、调用模式、关联用例、自然语言预期判定要点等）。在已启动 `app.py` 的前提下：

```bash
cd formation_sut
python run_experiment.py
python run_experiment.py --judge-config-name remote-scnet-deepseek
python run_experiment.py --llm-config-names local-lmstudio-gemma,remote-scnet-deepseek --judge-config-name remote-scnet-deepseek --repeat 3
```

完整参数说明见 `run_experiment.py` 文件末尾注释。运行结果默认输出到 `data/experiment_runs/<timestamp>/`，包含 `raw_results.jsonl`（原始记录）与 `summary.csv`（汇总表格）。

## 运行单元测试

```bash
cd formation_sut
pytest
```

## 安全提示

`config/llm_config.json`、`curl-test.txt` 等含真实密钥/Token 的文件已在 `.gitignore` 中排除，请勿手动提交。
