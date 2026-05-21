# app.py

import os              # ✅ add this line
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import requests

from pydantic import BaseModel
from typing import List

app = FastAPI(
    debug=True,           # ← this shows full traceback on 500
)
class FeedbackRequest(BaseModel):
    scenarioId: str
    category: str
    title: str
    prompt: str
    description: str
    explanation: str
    userResponse: str


class FeedbackResult(BaseModel):
    score: int
    shortFeedback: str
    detailedFeedback: str
    tips: List[str]

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def get_llm_feedback(category: str, prompt: str, user_response: str):
    system_content = (
        "You are a strict JSON-only coach. "
        "Return only valid JSON with keys: score, shortFeedback, detailedFeedback, tips."
    )

    user_content = f"""
Category: {category}
Prompt: {prompt}
User response: {user_response}
""".strip()

    payload = {
        "model": "meta-llama/llama-3.2-3b-instruct:free",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ai-backend-qhyv.onrender.com", 
        "X-OpenRouter-Title": "Conversation Coach",
    }

    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
        timeout=20,
    )
    response.raise_for_status()

    data = response.json()
    ai_text = data["choices"][0]["message"]["content"]
    return json.loads(ai_text)


@app.post("/feedback", response_model=FeedbackResult)
def get_feedback(request: FeedbackRequest):
    raw = get_llm_feedback(
    category=request.category,
    prompt=request.prompt,
    user_response=request.userResponse,
    )

    return FeedbackResult(
        score=int(raw["score"]),
        shortFeedback=raw["shortFeedback"],
        detailedFeedback=raw["detailedFeedback"],
        tips=raw["tips"],
    )