from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.api.routes import router
from src.dashboard.app import mount_dashboard

app = FastAPI(
    title="llm-eval-bench",
    description="Statistically rigorous LLM evaluation API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
mount_dashboard(app)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/dashboard")
