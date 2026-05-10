## 技术方案

- 文档解析：`PyMuPDF`、`python-docx`、`python-pptx`、`RapidOCR`（需要改进）
- 向量检索：`Chroma`
- 本地向量模型：`sentence-transformers + BAAI/bge-small-zh-v1.5`
- 图数据库：`Neo4j`
- 聊天大模型：兼容 OpenAI 风格接口
- 界面：`Streamlit`


## 准备数据

创建data文件夹

创建model文件夹


把你的资料放到 `data/` 目录下，例如：

- `pdf`
- `docx`
- `doc`
- `pptx`
- `png`
- `jpg`


## 环境变量

复制 `.env.example` 为 `.env`，至少填写：

```env
# 推荐：逗号分隔多个 Key，配额耗尽自动轮换（兼容旧写法 OPENAI_API_KEY）
OPENAI_API_KEYS=sk-key1,sk-key2
OPENAI_BASE_URL=https://api.openai.com/v1
# 推荐：逗号分隔多个模型，当前模型不可用时自动降级（兼容旧写法 OPENAI_CHAT_MODEL）
OPENAI_CHAT_MODELS=your-chat-model
EMBEDDING_PROVIDER=local
LOCAL_EMBED_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DEVICE=cpu
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
# 向量库分批写入大小（控制内存峰值，默认 500）
VECTOR_BATCH_SIZE=500
```

> **多 Key / 多模型说明**：`OPENAI_API_KEYS` 和 `OPENAI_CHAT_MODELS` 用英文逗号分隔。故障转移顺序为"同 Key 内先换模型，所有模型耗尽再换 Key"。旧写法 `OPENAI_API_KEY` 和 `OPENAI_CHAT_MODEL` 仍然兼容。

## 运行方式

先安装项目：

```bash
pip install -e .
```

说明：

- 首次运行会自动下载本地 embedding 模型
- 如果你有 NVIDIA GPU，可把 `.env` 里的 `EMBEDDING_DEVICE=cpu` 改成 `cuda`
- 运行前请先确保本地 `Neo4j` 已启动，且 `.env` 中密码正确


先启动neo4j数据库
```bash
neo4j.bat console
```


构建索引（向量库 + 知识图谱，自动跳过已处理的数据）：

```bash
kg-rag ingest
```

> **增量更新**：新增文件到 `data/` 后直接运行上述命令即可，已入库的 chunk 会自动跳过。中途崩溃后重跑也不会重复处理已完成的数据。

只构建向量库（不构建知识图谱，自动去重）：

```bash
kg-rag ingest-vector
```

重建向量库（删除旧数据，全新构建）：

```bash
kg-rag ingest-vector --reset
```

如果数据不在默认 `data/` 目录，也可以指定：

```bash
kg-rag ingest-vector --data-dir add_data
```

命令行提问：

```bash
kg-rag ask "什么是事务？"
```

查看调试上下文：

```bash
kg-rag debug "数据库范式有哪些？"
```

启动界面：

```bash
streamlit run app.py
```

## 本地 Qwen 测试

如果你想把知识图谱关系抽取改成本地模型，可以先测试本地 `Qwen` 推理。

当前项目提供了一个测试脚本，默认使用官方模型：

- `Qwen/Qwen3-4B-Instruct-2507`

模型权重会下载到：

- `kg_rag_demo/model/Qwen3-4B-Instruct-2507/`

先重新安装项目依赖：

```bash
pip install -e .
```

仅下载模型：

```bash
kg-rag-local-qwen-test --download-only
```

下载并执行一次本地推理：

```bash
kg-rag-local-qwen-test
```

自定义提示词：

```bash
kg-rag-local-qwen-test --prompt "请简要说明数据库系统的组成。"
```

如果你有 NVIDIA GPU，可以显式指定：

```bash
kg-rag-local-qwen-test --device cuda
```

如果只想在 CPU 上验证流程：

```bash
kg-rag-local-qwen-test --device cpu
```

## neo4j 基础指令


加载数据
```bash
neo4j-admin database load neo4j --from-path=导出图谱的路径
```

查询节点
```bash
match(n) return n
```


## TO DO

优化构建数据库部分

    1、优化文档解析方法

    2、优化向量库写入方法

    3、优化图谱生成方法

    4、在生成图谱时做一个实体表格

    5、图谱检索时加入实体链接增强（抽取实体后与实体表格做匹配）