# Backend Structure

当前后端的活跃实现已经收敛到 `app/services/`：

- `run.py`：Flask 启动入口
- `app/__init__.py`：Flask app 工厂
- `app/routes.py`：API 路由
- `app/services/config.py`：环境变量和运行配置
- `app/services/prompts.py`：4B / 8B 提示词
- `app/services/helpers.py`：通用工具函数
- `app/services/models.py`：本地模型加载和推理
- `app/services/kg_retriever.py`：Neo4j 检索
- `app/services/pipeline.py`：4B -> KG -> 8B 编排
- `app/services/__init__.py`：统一导出入口
- `demo_local_qwen_pipeline.py`：联调脚本

当前对外接口：
- `GET /api/health`
- `GET /api/diagnose/health`
- `POST /api/diagnose`
- `GET /api/diagnose/local-qwen-pipeline/health`
- `POST /api/diagnose/local-qwen-pipeline`

说明：
- `app/services/local_qwen/` 现在只作为兼容目录保留，主代码不再从那里加载。
- `routes.py` 和 `demo_local_qwen_pipeline.py` 都已经直接接到 `app.services`。
- 目录里若还看到旧的 `chat_store.py`、`rag_service.py`、`kg_service.py`、`llm_service.py`、`build_kg.py`、`train_model.py`，它们不在当前运行路径里。
