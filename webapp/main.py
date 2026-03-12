
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from webapp.api.routers import conversation

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(conversation.router)

if __name__ == "__main__":
	import uvicorn
	uvicorn.run(
		"webapp.main:app",
		host="0.0.0.0",
		port=8000,
		reload=True
	)
