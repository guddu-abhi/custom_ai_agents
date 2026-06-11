from pydantic import BaseModel


# Define request and response schemas
class ConversationRequest(BaseModel):
    user_id: str
    session_id: str | None = None
    message: str


class OttoRequest(BaseModel):
    query: str
