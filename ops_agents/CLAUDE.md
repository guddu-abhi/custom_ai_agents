# ops_agents/ — OpenAI Agents SDK specialist agents

Specialist agents handed-off to from `customer_desk_agent` (the triage router,
defined in `customer_triage_agent.py`): `product_advisor`, `billing`, `account`,
`post_order_support`. Each agent declares its tools, instructions, and output
type.

## Conventions
- One agent per file: `<role>_agent.py`. Module-level `<role>_agent = Agent(...)`.
- Register every agent in `registry.py:AGENT_REGISTRY` so handoff/lookup works.
  Registry keys are the public agent name (e.g. `"ProductAdvisorAgent"`) — the
  same strings agents return in `agent_handed_off_to`.
- Output type: `ops_agents.model.base_response.AgentsBaseResponse` (or a
  subclass). The response includes the `agent_handed_off_to` field webapp uses
  to update `USER_AGENT_STATE`.
- Tools come from `tools/` — agents wire them in, they don't define them.
- For grounded product answers, tools should call
  `retrieval.SearchService` + `generation.GenerationService`; agents must not
  call the LLM directly outside the Agents SDK runtime.
- Model name (`gpt-5-nano`, `gpt-5-mini`, etc.) as the Agent's `model=...` arg.
- Conversation memory uses `agents.extensions.memory.SQLAlchemySession`,
  constructed per-session in `utils.session_utils.get_user_session`.

## Don't
- Don't put tool implementations here — those belong in `tools/`.
- Don't put HTTP / FastAPI code here — webapp wraps agents, not the other way.
- Don't open DB connections inside agent definitions — tools handle data
  access; agents orchestrate.
- Don't hardcode user/session IDs — they come from `webapp` request context.
- Don't bypass the registry: handoffs by name go through `get_agent_by_name`.
- Don't add a new specialist without also adding its entry to `AGENT_REGISTRY`
  and to the triage agent's routing rules — handoff resolution silently returns
  `None` from the registry otherwise.
