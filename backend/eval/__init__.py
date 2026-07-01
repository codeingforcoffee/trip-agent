"""离线评测 harness（M8）。

核心难题：怎么**离线、可复现**地评一个**非确定性**的 Agent？本包的答案分三招：
  1. 两层打分——能确定性断言的绝不用 LLM（工具轨迹/检索命中/audit 事件/步数/PII），
     只有开放式质量（答得对不对、忠不忠于证据）才交 LLM-as-judge（judge.py）。
  2. 录制回放（replay.py）——把唯一"非确定 + 联网 + 花钱"的依赖（LLM）用 cassette 冻结：
     `make eval` 回放（agent + judge 全回放）→ 完全离线、免费、秒级、CI 友好；
     `make eval-live` 真跑 DeepSeek 评当前 prompt 下的真实质量，`make eval-record` 重录。
     图 `build_graph(llm,...)` 是依赖注入的——把 ReplayLLM/RecordingLLM 塞进去即可，M1 的回报。
  3. 黄金数据集（datasets/*.jsonl）——每条场景带期望工具轨迹 / rubric / RAG ground-truth /
     期望 audit，覆盖下单链路 / RAG / 安全护栏 / 工具选择+步数 / 记忆召回 五类。

关键洞察（面试金句）：把两件根本不同的事分开——**确定性组件的回归**（工具/检索/护栏/路由）
用 cassette 回放；**答案质量在新 prompt 下的评测**必须真跑 LLM（拿旧输出评新 prompt 是自欺）。

cassette 用**按场景有序序列**回放（不做消息内容哈希键）：同一场景在回放下控制流可复现 →
LLM 调用次数与顺序稳定 → 逐次 pop 即可。天然免疫 system prompt 里 {today}、记忆前缀等易变内容。
"""
