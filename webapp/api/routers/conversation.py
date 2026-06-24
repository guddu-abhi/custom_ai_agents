import json
from uuid import uuid4

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from agents import Runner, TResponseInputItem
from otto_lib.db.session import get_db_session, get_user_session
from otto_lib.logging import get_logger
from webapp.schema.api_request import ConversationRequest

logger = get_logger()
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
