from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from httpcore import request
from pydantic import BaseModel
from agents import Runner, TResponseInputItem
from ops_agents.product_advisor_agent import product_advisor_agent
from ops_agents.billing_agent import billing_agent
from utils.session_utils import get_db_session, get_user_session, db_session
from webapp.schema.api_request import ConversationRequest
from webapp.schema.api_response import ConversationResponse
from ops_agents.customer_triage_agent import customer_desk_agent
from ops_agents.registry import get_agent_by_name
from domain.models.conversation_state import MinimalConversation, UserMessage, AssistantMessage
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, base
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import json
from sqlalchemy.orm import declarative_base
from loguru import logger

router = APIRouter()


# Agent state: {(user_id, session_id): agent_name}
USER_AGENT_STATE: dict[tuple[str, str], str] = {}


async def sse(user_id: str, active_agent, input_items, session_id: str, session: Session):

    try:
        logger.info(f"Session {session_id}: starting conversation stream with agent '{active_agent.name}'")
        stream = Runner.run_streamed(
            active_agent,
            input_items,
            session=session
        )

        async for event in stream.stream_events():

            match event.type:

                case "raw_response_event":

                    if event.data.type == "response.output_text.delta":

                        yield f"data: {json.dumps({'type': 'output', 'delta': event.data.delta})}\n\n"

                case "run_item_stream_event":

                    if event.name in ("handoff_requested", "handoff_occured"):
                        logger.info(f"Session {session_id}: handoff event '{event.name}'")

        # STREAM FINISHED HERE

        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
        logger.info(f"Session {session_id}: conversation stream completed")
        logger.info(f"final_output: {stream.final_output}")
        if stream.final_output is not None:
            if stream.final_output.agent_handed_off_to is not None:
                # update agent state for this session
                USER_AGENT_STATE[(user_id, session_id)] = stream.final_output.agent_handed_off_to

        logger.info(f"Structured output: {stream.final_output}")

    except Exception as e:

        error_msg = json.dumps({'type': 'error', 'message': str(e)})
        yield f"data: {error_msg}\n\n"

@router.post("/chat/conversation")
async def chat_conversation(
    request: ConversationRequest,
    session: Session = Depends(get_db_session)
):
    # Use provided session_id or generate a new one
    session_id = request.session_id or str(uuid4())

    # Get database session (will create user_session record if it doesn't exist)
    # async with db_session() as session:
    user_session = await get_user_session(request.user_id, session_id, session)

    logger.info(f"Current agent state for user {request.user_id}, session {session_id}: {USER_AGENT_STATE.get((request.user_id, session_id))}")
    # Resolve the active agent from state, defaulting to CustomerDeskAgent for new sessions
    agent_name = USER_AGENT_STATE.get((request.user_id, session_id), "CustomerDeskAgent")
    active_agent = get_agent_by_name(agent_name)
        
    # Add the new user message
    input_item: list[TResponseInputItem] = [{
        "role": "user",
        "content": request.message
    }]

    return StreamingResponse(
        sse(
            request.user_id,
            active_agent,
            input_item,
            session_id,
            user_session,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
