"""user_session table — user_id <-> session_id registry for conversations

The webapp passes (user_id, session_id) on every conversational turn (like a
cookie). This table records the pairing; the SDK's own `agent_sessions` /
`agent_messages` tables (auto-created by SQLAlchemySession) hold the actual
messages, keyed by session_id. `utils.session_utils.get_user_session` looks up /
inserts rows here before returning the SDK-backed session.

Revision ID: f1a2b3c4d5e6
Revises: e7b4d2c8a915
Create Date: 2026-06-10 12:00:00.000000

"""
from collections.abc import Sequence
from typing import Union

from alembic import op

revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "e7b4d2c8a915"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # session_id is the primary key: it is globally unique (it keys the SDK's
    # agent_sessions store) and a session belongs to exactly one user, so this
    # also prevents one user_id from claiming another user's session_id.
    op.execute(
        """
        CREATE TABLE catalog.user_session (
            session_id  TEXT        PRIMARY KEY,
            user_id     TEXT        NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    # Supports "list all sessions for a user".
    op.execute(
        "CREATE INDEX ix_user_session_user_id ON catalog.user_session (user_id)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS catalog.user_session")
