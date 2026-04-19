# 基于临床表征图谱与大模型推理的疾病辅助诊断系统

本项目实现了一条完整的医疗辅助诊断链路：

用户输入症状与病史  
`->` 轻量模型做医疗节点识别/归一化  
`->` Neo4j 知识图谱检索核心医学事实  
`->` 大模型在图谱约束下生成候选疾病、检查、就诊科室与处理建议  
`->` 前端单页报告展示结果

项目当前重点解决的是两个问题：

- 将用户口语化、模糊化、不完整的问诊表达映射到知识图谱节点
- 将图谱证据与大模型生成结果整合成可展示、可追溯的诊断报告

## 1. 项目架构

系统主链路如下：

```text
用户输入
  ↓
Qwen 4B（节点识别 / grounding）
  ↓
Neo4j 知识图谱检索
  ↓
Qwen 8B（受图谱约束的建议生成）
  ↓
前端单页报告
```

当前后端已经按模块拆分，核心职责如下：

- `backend/app/services/models.py`
  负责本地 Qwen 模型加载与推理
- `backend/app/services/kg_retriever.py`
  负责 Neo4j 检索、候选疾病聚合、证据路径提取
- `backend/app/services/pipeline.py`
  负责 `4B -> KG -> 8B` 整体编排
- `backend/app/routes.py`
  负责 HTTP 接口

## 2. 仓库结构

```text
med_llm/
├─ frontend/                      # React + Vite 前端
├─ backend/                       # Flask 后端与本地模型调用链路
├─ algorithm/                     # 数据集构建、训练、知识图谱脚本
│  ├─ data/
│  │  ├─ raw/                     # 原始数据
│  │  ├─ processed/               # 中间结果、统计、图谱离线产物
│  │  └─ llamafactory/            # LLaMA-Factory 训练数据
│  ├─ kg/                         # Neo4j 导入与图谱导出脚本
│  └─ scripts/                    # 数据集构建与训练脚本
├─ LLaMA-Factory/                 # LLaMA-Factory 数据目录
└─ scripts/                       # Linux 启停脚本
```

## 3. 关键能力

### 3.1 Query 到图谱节点映射

该任务不是传统 BIO NER，而是更偏向 `query grounding`：

- 输入是用户真实问诊口吻
- 输出是知识图谱中的标准节点
- 重点识别：
  - `symptom`
  - `disease`
  - `department`
  - `check`
  - `drug`

例如：

```json
{
  "nodes": [
    {"mention": "发高烧", "node_name": "发热", "node_type": "symptom"},
    {"mention": "浑身疼", "node_name": "全身疼痛", "node_type": "symptom"}
  ]
}
```

### 3.2 图谱约束生成

8B 模型不直接“凭空回答”，而是使用图谱检索结果生成结构化输出，核心字段包括：

- 识别出的症状节点
- 候选疾病
- 推荐科室
- 推荐检查
- 药物 / 治疗方式
- 饮食建议 / 禁忌
- 图谱证据路径

### 3.3 同账号双图谱

当前使用 Neo4j Community。Community 版不能在同一实例下像 Enterprise 那样自由使用多标准数据库，因此项目采用 `graph_name` 命名空间方案，在同一个账号、同一个数据库中同时维护：

- `common`
- `full`

即：

- 节点唯一键：`(graph_name, name)`
- 关系也带 `graph_name`
- 导入与清理时只作用于指定 `graph_name`

## 4. 原始数据

当前主要使用以下原始数据：

- `algorithm/data/raw/medical.json`
  全量医疗百科结构化数据，作为知识图谱源数据
- `algorithm/data/raw/imcs21/IMCS-V2_train.json`
- `algorithm/data/raw/imcs21/IMCS-V2_dev.json`
- `algorithm/data/raw/imcs21/IMCS-V2_test.json`
- `algorithm/data/raw/imcs21/symptom_norm.csv`
  用于口语症状、标准症状之间的映射
- `algorithm/data/raw/train_0001_of_0001.json`
  中文医疗对话数据，用于补充真实用户问诊表达

## 5. 数据集构建

项目当前有两类主要数据集构建脚本。

### 5.1 常见病范围数据集

用于常见病原型或答辩展示：

```powershell
conda activate med
cd D:\Yudou\med_llm
python .\algorithm\scripts\build_query2graph_common_dataset.py --target-size 10000
```

常见输出目录示例：

- `algorithm/data/processed/common_disease/query2graph_annotations.json`
- `algorithm/data/processed/common_disease/query2graph.json`

### 5.2 全量图谱覆盖数据集

用于尽可能覆盖 `medical.json` 全量节点空间：

```powershell
conda activate med
cd D:\Yudou\med_llm
python .\algorithm\scripts\build_query2graph_full_dataset.py
```

默认输出：

- `algorithm/data/processed/query2graph_full_annotations.json`
- `algorithm/data/llamafactory/query2graph_full_sharegpt.json`
- `LLaMA-Factory/data/query2graph_full_sharegpt.json`
- `algorithm/data/processed/query2graph_full_stats.json`
- `algorithm/data/processed/medical_full_kg_source.jsonl`

如果希望自动注册到 LLaMA-Factory：

```powershell
python .\algorithm\scripts\build_query2graph_full_dataset.py --register-dataset
```

## 6. 知识图谱构建

### 6.1 导出离线图谱产物

导出 `common` 图谱：

```powershell
python .\algorithm\kg\export_common_kg_artifacts.py `
  --data-path .\algorithm\data\processed\common_disease\medical_common_subset.json `
  --graph-name common
```

导出 `full` 图谱：

```powershell
python .\algorithm\kg\export_common_kg_artifacts.py `
  --data-path .\algorithm\data\processed\full_disease\medical_full_kg_source.jsonl `
  --graph-name full
```

默认会生成：

- `nodes.csv`
- `edges.csv`
- `graph.json`
- `summary.json`

### 6.2 导入 Neo4j

导入 `common`：

```powershell
python .\algorithm\kg\build_kg.py `
  --data-path .\algorithm\data\processed\common_disease\medical_common_subset.json `
  --graph-name common `
  --neo4j-uri bolt://localhost:7687 `
  --neo4j-user neo4j `
  --neo4j-password 你的密码 `
  --clear-existing
```

导入 `full`：

```powershell
python .\algorithm\kg\build_kg.py `
  --data-path .\algorithm\data\processed\full_disease\medical_full_kg_source.jsonl `
  --graph-name full `
  --neo4j-uri bolt://localhost:7687 `
  --neo4j-user neo4j `
  --neo4j-password 你的密码 `
  --clear-existing
```

说明：

- `--clear-existing` 只会清空当前 `graph_name` 对应的子图，不会删另一套图谱
- 如果你是旧版本图谱，脚本会自动处理旧的 name-only 约束

## 7. 模型训练

项目训练主线基于 LLaMA-Factory。

### 7.1 4B 节点识别模型

目标：

- 将用户 query 映射到图谱节点

训练数据：

- 常见病 `query2graph`
- 全量覆盖 `query2graph_full_sharegpt`

### 7.2 9B 建议生成模型

目标：

- 根据用户 query + 图谱检索结果输出最终建议

相关脚本：

- `algorithm/scripts/prepare_graph_advice_dataset.py`
- `algorithm/scripts/prepare_meddialog_advice_dataset.py`

如果你已经在 `LLaMA-Factory/data/dataset_info.json` 中注册数据集，可以直接用对应 YAML 训练。

## 8. 后端启动

开发模式：

```powershell
conda activate med_llm
cd D:\Yudou\med_llm\backend
python .\run.py
```

默认端口：

- `5000`

关键接口：

- `GET /api/health`
- `GET /api/diagnose/health`
- `POST /api/diagnose`
- `GET /api/diagnose/local-qwen-pipeline/health`
- `POST /api/diagnose/local-qwen-pipeline`

## 9. 前端启动

开发模式：

```powershell
conda activate med
cd D:\Yudou\med_llm\frontend
npm install
npm run dev
```

默认端口：

- `5173`

## 10. Linux 服务器启动

项目提供了 Linux 启停脚本：

- `scripts/start.sh`
- `scripts/stop.sh`

启动：

```bash
bash scripts/start.sh
```

停止：

```bash
bash scripts/stop.sh
```

当前脚本会统一管理：

- Neo4j
- 后端
- 前端

并保证：

- 任一服务启动失败，则整体回滚
- 关闭终端后进程继续运行

## 11. 配置说明

主要配置位于：

- `backend/app/services/config.py`

包含：

- 4B 模型路径
- 8B 模型路径
- Neo4j 连接信息
- 设备配置

如果你不想通过环境变量传模型路径，直接改这个文件即可。

## 12. 推荐调试顺序

建议按下面顺序检查：

1. 先确认 Neo4j 可连接
2. 再确认 `common/full` 图谱已导入
3. 单独测试 4B 节点识别输出
4. 单独测试图谱检索结果
5. 最后联调 8B 生成和前端展示

## 13. 当前已知注意点

- Neo4j Community 不支持像 Enterprise 那样直接用多标准数据库，所以本项目采用 `graph_name` 命名空间方案。
- 同一 Neo4j 中如果已存在旧版只按 `name` 唯一的约束，需要先用新版 `build_kg.py` 重新导入。
- 如果同时导入 `common` 和 `full`，后端检索层也应按 `graph_name` 过滤，否则会混查。

## 14. 后续可继续完善的方向

- 给 `kg_retriever.py` 增加 `graph_name` 选择
- 为 4B 节点识别增加更系统的评测集
- 为 8B 建议生成增加结构化自动评测
- 增加图谱节点同义词表与人工校正规则表



