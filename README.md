# med_llm

基于临床知识图谱与大模型推理的医疗辅助诊断系统。

项目将本地大模型、NER 结构化识别和 Neo4j 知识图谱检索结合，构建一个面向症状问诊、候选疾病推荐、检查与科室建议的辅助诊断后台。前端使用 Vite+React，后端使用 Flask 提供 API 服务。

## 目录结构

- `backend/`：Flask 后端服务、API 路由、模型与图谱检索逻辑
- `frontend/`：Vite + React 前端应用
- `algorithm/`：数据准备、训练脚本、知识图谱构建与本地冒烟验证
- `Qwen3-4B-Med_ft/`：默认 NER 模型目录示例；实际模型路径可通过环境变量配置

## 核心能力

1. 通过本地 NER 模型将用户问诊文本转换为结构化实体
2. 使用 Neo4j 知识图谱检索相关疾病、检查、科室、药物等信息
3. 基于图谱结果和推理模型生成辅助诊断建议
4. 提供标准 API 接口，支持前端和外部服务调用

## 快速开始

### 依赖安装

```powershell
pip install -r requirements.txt
```

前端开发需要安装 Node.js:

```powershell
cd frontend
npm install
```

### 启动后端

后端主入口文件：`backend/run.py`

```powershell
set MED_NER_MODEL_PATH=../Qwen3-4B-Med_ft
set MED_REASON_MODEL_PATH=../Qwen3-8B-Instruct
set NEO4J_URI=bolt://localhost:7687
set NEO4J_USER=neo4j
set NEO4J_PASSWORD=your_password
python backend/run.py
```

或使用可选入口：

```powershell
python backend/serve.py --host 0.0.0.0 --port 5000
```

### 启动前端

```powershell
cd frontend
npm run dev
```

如果希望后端直接托管前端静态资源：

```powershell
cd frontend
npm run build
set MED_SERVE_FRONTEND_DIST=1
python backend/run.py
```

## 后端配置

可通过环境变量覆盖默认值：

- `MED_NER_MODEL_PATH`：NER 模型目录路径
- `MED_REASON_MODEL_PATH`：诊断推理模型目录路径
- `NEO4J_URI`：Neo4j 连接 URI，默认 `bolt://localhost:7687`
- `NEO4J_USER`：Neo4j 用户名，默认 `neo4j`
- `NEO4J_PASSWORD`：Neo4j 密码
- `MED_MODEL_DEVICE`：模型设备，默认 `auto`
- `MED_SERVE_FRONTEND_DIST`：启用后端静态前端托管，值为 `1`

## API 接口

- `GET /api/health`
- `GET /api/diagnose/health`
- `POST /api/diagnose`
- `POST /api/diagnose/local-qwen-pipeline`
- `GET /api/diagnose/local-qwen-pipeline/health`

示例请求体：

```json
{
  "query": "患者主诉头晕、乏力，伴随心悸",
  "history": "",
  "limit": 5
}
```

## 运行原理

后端核心由 `backend/app/services/pipeline.py` 实现：

- 先用本地 NER 模型抽取症状、疾病、科室、检查项等节点
- 再用 `backend/app/services/kg_retriever.py` 向 Neo4j 提取图谱候选疾病与相关路径
- 最后用推理模型生成结构化辅助诊断结果，并合并图谱证据

## 算法与训练

`algorithm/` 目录包含训练与数据准备脚本：

- `algorithm/scripts/prepare_graph_advice_dataset.py`
- `algorithm/scripts/prepare_meddialog_advice_dataset.py`
- `algorithm/scripts/train_qwen35_2b_ner.ps1`
- `algorithm/scripts/train_qwen35_9b_advice.ps1`

更多训练说明请查看 `algorithm/README.md`。

## 注意事项

- 本项目为辅助诊断系统，不能替代专业医生诊断
- 需要本地 Neo4j 数据库和可用模型权重
- 运行模型时依赖 `torch`、`transformers`、`neo4j` 等库
- 默认模型路径并不包含在仓库中，请根据实际环境配置

## 贡献与扩展

- 可扩展图谱节点类型和 Neo4j 查询逻辑
- 可替换本地推理模型为更高性能的模型或服务
- 可集成更多临床数据源和诊断流程优化

