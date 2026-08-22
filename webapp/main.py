import os

from dotenv import load_dotenv

# Load root .env into the process env BEFORE importing the Agents SDK / routers,
# so the OpenAI SDK and its tracing exporter can read OPENAI_API_KEY themselves.
load_dotenv()

import mlflow  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from webapp.api.routers import otto  # noqa: E402


def _init_mlflow() -> None:
    """Enable MLflow tracing at import time.

    Must run at module scope (not in the ``__main__`` block): the app is served
    via ``uvicorn webapp.main:app``, so ``__name__`` is ``"webapp.main"`` in the
    worker and a ``__main__``-guarded setup would never run where requests are
    handled. ``mlflow.openai.autolog()`` instruments the OpenAI Agents SDK
    (``Runner``) used by the otto agents. Default backend is the local SQLite
    ``mlflow.db``; traces need a DB backend (the ``./mlruns`` file store is
    deprecated and does not render traces).
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "Trace Otto Agent"))
    mlflow.openai.autolog()


_init_mlflow()

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(otto.router)

if __name__ == "__main__":
	import uvicorn
	uvicorn.run(
			"webapp.main:app",
			host="0.0.0.0",
			port=8000,
			reload=True
		)
