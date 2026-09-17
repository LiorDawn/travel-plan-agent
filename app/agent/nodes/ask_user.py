"""外层 ask_user 节点 — 缺参时挂起外层图，向前端透传追问（对应重构文档 §9.1/§9.2）

唯一挂起点，服务两种来源（用 state.interrupt_source 区分，resume 后由 route_after_ask_user 回跳）：
  - plan 缺参：一次性追问所有缺失字段 → resume 回 plan 重判
  - 子图缺参：透传 subagent_interrupt → resume 回 planner_run 续跑子图
首访 interrupt() 挂起 → graph.run() 读到 interrupt → 发 ask_user + done(need_input)；
resume 时 interrupt() 返回用户补充 → 写入 resume_input。
"""
from langgraph.types import interrupt

from app.agent.state import AgentState
from app.core.logging import logger


def _build_payload(state: AgentState) -> dict:
    """统一澄清协议载荷：{source, round, message, questions:[...]}。

    source 三种：
      - need_info：模型语义自评缺信息 → 把 needs 转成 questions（含 meta.key / meta.required）
      - plan：机械缺参每个字段 → 一条 text 型 question（含 meta: tool/field）
      - subagent：透传子图中断载荷
    """
    source = state.get("interrupt_source", "plan")
    if source == "subagent":
        payload = state.get("subagent_interrupt") or {}
        if isinstance(payload, dict) and "message" not in payload:
            payload = {"message": str(payload)}
        payload.setdefault("source", "subagent")
        return payload

    if source == "need_info":
        needs = state.get("needs") or []
        questions = []
        for i, n in enumerate(needs, start=1):
            questions.append({
                "id": f"q{i}",
                "question": n.get("question", ""),
                "type": n.get("type", "text"),          # text/number/date/option
                "options": n.get("options") or [],
                "required": n.get("required", True),    # False=可选偏好，前端可标"选填"
                "hint": "",
                "meta": {"key": n.get("key", ""), "required": n.get("required", True)},  # 回灌用
            })
        return {
            "source": "need_info",
            "round": 1,
            "message": "；".join(n.get("question", "") for n in needs) or "还需补充一些信息。",
            "questions": questions,
        }

    missing = state.get("missing_params", [])
    questions = []
    for i, m in enumerate(missing, start=1):
        tool, fields = m.get("tool", ""), m.get("fields", [])
        questions.append({
            "id": f"q{i}",
            "question": f"规划「{tool}」还需补充：{', '.join(fields)}",
            "type": "text",
            "options": [],
            "required": True,
            "hint": "",
            # 附加元数据，前端可据此做结构化输入/回填
            "meta": {"tool": tool, "missing_fields": fields},
        })
    return {
        "source": "plan",
        "round": 1,
        "message": "；".join(
            f"规划「{m['tool']}」还需补充：{', '.join(m.get('fields', []))}" for m in missing
        ) or "还需补充必要参数。",
        "questions": questions,
    }


async def ask_user_node(state: AgentState) -> AgentState:
    """把缺参信息作为外层 interrupt 载荷挂起；resume 后回填用户补充。

    resume 时把用户回答解析并回灌：
      - 前端按 ClarifyPayload.questions 收集、以 {"answers": {key: value}} JSON 返回 → 合并写 clarifications
        （key 取自 question.meta.key；checkbox/option 等纯文本 key 也按 meta.key 对齐），plan 据此重判。
      - 纯文本（子图自愈路径 / 自然语言补充） → 写 resume_input，供子图读取。
    """
    payload = _build_payload(state)
    logger.info("ask_user_pause", source=state.get("interrupt_source"), payload=payload.get("message"))
    user_reply = interrupt(payload)  # 首次挂起；resume 时返回用户补充
    reply_text = user_reply if isinstance(user_reply, str) else str(user_reply)
    logger.info("ask_user_resumed", got=reply_text[:60])
    # 追问计数 +1：让 router 的 "已追问过仍缺参 → 降级 summarize" 真正生效（防非 travel 无限打断）
    ask_count = (state.get("ask_count", 0) or 0) + 1
    new_state = {**state, "resume_input": reply_text, "ask_count": ask_count}
    # need_info 已弹过一次窗：标记防二次弹窗（router ② 依赖它跳过重复收集）
    if state.get("interrupt_source") == "need_info":
        new_state["need_info_asked"] = True
    # P0 修复：弹窗收集的回答要回灌给 plan，否则询问闭环断裂（用户白答一轮）。
    # 按 ClarifyPayload 的 meta.key 把 answers 合并进 clarifications；plan.py 重判时读取。
    parsed = _parse_answers(reply_text)
    if parsed:
        clar = dict(state.get("clarifications") or {})
        clar.update(parsed)
        new_state["clarifications"] = clar
    return new_state


def _parse_answers(reply_text: str) -> dict | None:
    """尝试把用户回答按 ClarifyPayload 协议解析为 {meta.key: answer}。

    支持两种形态：
      - 前端结构化：{"answers": {"budget": "8000", "days": "3"}} → 直接使用 answers
      - 紧凑 JSON 对象：{"budget": "8000"} → 直接使用；含非 answers/非标键则忽略
    非 JSON / 纯文本返回 None（走 resume_input 路径）。
    """
    import json
    text = reply_text.strip()
    if not text.startswith("{"):
        return None
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("answers"), dict):
        answers = obj["answers"]
    else:
        answers = {k: v for k, v in obj.items() if not k.startswith("_")}
    answers = {k: v for k, v in answers.items() if v is not None and v != ""}
    return answers or None