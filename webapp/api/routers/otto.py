import dataclasses
import json
import os
from uuid import uuid4

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from agents.memory import Session
from domain.models.search import OttoAnswer
from otto_lib.db.session import get_db_session, get_user_session
from ottoai.conversational_otto_agent import ConversationalOttoAgent
from ottoai.otto_agent import OttoAgent
from ottoai.single_agent import SingleAgentOtto
from webapp.schema.api_request import ConversationRequest, OttoRequest

router = APIRouter()

otto_agent = OttoAgent()
conversational_otto_agent = ConversationalOttoAgent()
single_agent_otto = SingleAgentOtto()

# Feature flag: when true, /chat/otto streams answer deltas over SSE; when false
# (default), it returns a single JSON body with the full answer and citations.
# Applies to both /chat/otto and /chat/otto/conversation.
SHOULD_STREAM_RESPONSE = os.getenv("SHOULD_STREAM_RESPONSE", "False").lower() == "true"

_SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive"}


async def sse(query: str):
    try:
        async for item in otto_agent.stream(query):
            if isinstance(item, OttoAnswer):
                citations = [dataclasses.asdict(c) for c in item.citations]
                yield f"data: {json.dumps({'type': 'done', 'citations': citations})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'output', 'delta': item})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


async def conversation_sse(session: Session, message: str, session_id: str):
    try:
        async for item in conversational_otto_agent.stream(session, message):
            if isinstance(item, OttoAnswer):
                citations = [dataclasses.asdict(c) for c in item.citations]
                yield (
                    "data: "
                    + json.dumps(
                        {"type": "done", "citations": citations, "session_id": session_id}
                    )
                    + "\n\n"
                )
            else:
                yield f"data: {json.dumps({'type': 'output', 'delta': item})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


async def single_agent_conversation_sse(session: Session, message: str, session_id: str):
    try:
        async for item in single_agent_otto.stream(session, message):
            if isinstance(item, OttoAnswer):
                citations = [dataclasses.asdict(c) for c in item.citations]
                yield (
                    "data: "
                    + json.dumps(
                        {"type": "done", "citations": citations, "session_id": session_id}
                    )
                    + "\n\n"
                )
            else:
                yield f"data: {json.dumps({'type': 'output', 'delta': item})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@router.post("/chat/otto/conversation/single_agent")
async def chat_otto_conversation_single_agent(
    request: ConversationRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Agentic multi-turn Otto. A single gpt-4o agent owns query planning AND
    answering, and calls the retriever as a function_tool when it sees fit. The
    SDK Session provides multi-turn memory (messages + tool calls/results), so
    the agent can answer follow-ups from earlier results or re-search on its own.
    Same SSE / JSON response shape as /chat/otto/conversation."""
    session_id = request.session_id or str(uuid4())
    store = await get_user_session(request.user_id, session_id, db)

    if SHOULD_STREAM_RESPONSE:
        return StreamingResponse(
            single_agent_conversation_sse(store, request.message, session_id),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    answer = await single_agent_otto.run(store, request.message)
    return {
        "answer": answer.answer,
        "citations": [dataclasses.asdict(c) for c in answer.citations],
        "session_id": session_id,
    }
