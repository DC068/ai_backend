# File: app.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
import json


app = FastAPI()


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


def get_llm_feedback(category: str, prompt: str, user_response: str) -> Optional[dict]:
    """
    Call OpenRouter with a structured prompt that returns JSON feedback.
    """
    system_content = (
        "You are an emotional intelligence / critical thinking coach.\n"
        "You will score a user response out of 100, give a short summary, "
        "a detailed feedback paragraph, and 3–5 tips.\n"
        "RETURN A JSON object like this (no extra text):\n"
        "{\n"
        '  "score": 80,\n'
        '  "shortFeedback": "Good awareness shown.",\n'
        '  "detailedFeedback": "You acknowledged their feelings...",\n'
        '  "tips": ["Tip 1", "Tip 2"]\n'
        "}"
    )

    user_content = f"""
Category: {category}
Prompt: {prompt}
User response: {user_response}

Please score this response out of 100 and give structured feedback as JSON.
""".strip()

    payload = {
        "model": "openrouter/free",    # or pick one like "meta-llama/llama-3.2-3b-instruct:free"
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Optional: OpenRouter requests a referrer / title for rankings
        "HTTP-Referer": "https://ai-backend-qhyv.onrender.com",
        "X-Title": "Conversation Coach Backend",
    }

    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=10,
        )
        if response.status_code != 200:
            raise Exception(f"OpenRouter error: {response.status_code}")

        data = response.json()
        if "choices" not in data or not data["choices"]:
            raise Exception("No choices in OpenRouter response")

        ai_text = data["choices"][0]["message"]["content"]

        # In real code, parse this JSON instead of using hardcoded fallback
        # but this keeps the backend running even if parsing fails
        try:
            parsed = json.loads(ai_text)
            score = min(max(0, int(parsed.get("score", 50))), 100)
            short = parsed.get("shortFeedback", "Good start.")
            detailed = parsed.get("detailedFeedback", "Your response was clear and on topic.")
            tips = parsed.get("tips", [
                "Be more specific about what you would actually say or do.",
                "Focus on the other person's feelings or logical structure.",
            ])
            return {
                "score": score,
                "shortFeedback": short,
                "detailedFeedback": detailed,
                "tips": tips,
            }
        except json.JSONDecodeError:
            pass

    except Exception as e:
        print(f"LLM call failed: {e}")

    # Fallback in case of error or bad JSON
    return {
        "score": 50,
        "shortFeedback": "The AI feedback service is temporarily unavailable.",
        "detailedFeedback": f"Service error: {e}" if 'e' in locals() else "Please try again later.",
        "tips": [
            "Keep your responses clear and direct.",
            "Make sure you directly answer the question in the prompt.",
        ],
    }


@app.post("/feedback", response_model=FeedbackResult)
def get_feedback(request: FeedbackRequest):
    if not OPENROUTER_API_KEY:
        # Fallback when the env var is missing
        base_score = 50 + (len(request.userResponse) // 20)
        base_score = max(30, min(100, base_score))
        return FeedbackResult(
            score=base_score,
            shortFeedback="Feedback is currently running in offline mode.",
            detailedFeedback="Your response is being evaluated based on length and structure.",
            tips=[
                "Try making your answer more specific.",
                "Align your response with the scenario's goal.",
            ],
        )

    raw = get_llm_feedback(
        category=request.category,
        prompt=request.prompt,
        userResponse=request.userResponse,
    )

    return FeedbackResult(
        score=raw["score"],
        shortFeedback=raw["shortFeedback"],
        detailedFeedback=raw["detailedFeedback"],
        tips=raw["tips"],
    )