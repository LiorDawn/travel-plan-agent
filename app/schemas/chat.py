from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    user_id: str = Field(default="anonymous", description="用户ID")
    conversation_id: str = Field(default="", description="会话ID，为空则新建")
    resume: bool = Field(default=False, description="是否为续跑模式（从 interrupt 断点恢复）")
    thinking_mode: bool = Field(default=False, description="思考开关：False=快查(execute)，True=深度规划(react 子图)")
