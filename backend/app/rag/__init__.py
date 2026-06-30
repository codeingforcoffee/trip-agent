"""RAG 子系统（M5）：写路径 ingest / 读路径 retriever / 切块 chunk。

policy 工具从"关键词匹配"升级为"向量检索 + 引用"就发生在这一层之上。
所有检索都按 tenant_id 前过滤（多租户向量隔离）。
"""
