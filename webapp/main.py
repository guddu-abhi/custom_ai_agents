from dotenv import load_dotenv

# Load root .env into the process env BEFORE importing the Agents SDK / routers,
# so the OpenAI SDK and its tracing exporter can read OPENAI_API_KEY themselves.
load_dotenv()

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from webapp.api.routers import otto  # noqa: E402

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
