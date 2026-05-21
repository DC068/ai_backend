# app.py

import os
import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional


app = FastAPI(debug=True)


# === Models ===

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


# === Env + OpenRouter setup ===

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


# === Safe LLM feedback function (handles 429, timeouts, bad JSON) ===

def get_llm_feedback(category: str, prompt: str, user_response: str) -> dict:
    """
    Call OpenRouter and return a dict with:
        {"score": int, "shortFeedback": str, "detailedFeedback": str, "tips": [...]}
    If OpenRouter fails (429, timeout, bad JSON, etc.), return a fallback.
    """
    if not OPENROUTER_API_KEY:
        return {
            "score": 50,
            "shortFeedback": "No AI backend key configured.",
            "detailedFeedback": "Responses are based on length and structure only.",
            "tips": [
                "Service is running in fallback mode while AI is unavailable.",
                "Keep your answers clear and direct.",
            ],
        }

    system_content = (
        "You are a strict JSON‑only coach.\n"
        "Return only valid JSON with these keys:\n"
        "{\n"
        '  "score": integer between 0 and 100,\n'
        '  "shortFeedback": string,\n'
        '  "detailedFeedback": string,\n'
        '  "tips": list of strings\n'
        "}\n"
        "Do not include any other text before or after the JSON."
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

    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=20,
        )

        # Handle 429 gracefully (rate limit)
        if response.status_code == 429:
            print("OpenRouter 429 Too Many Requests; using fallback response.")
            return {
                "score": 50,
                "shortFeedback": "Too many requests right now.",
                "detailedFeedback": "The AI service is temporarily rate‑limited. You can keep practicing your responses even without AI feedback.",
                "tips": [
                    "Wait a few seconds before asking for more feedback.",
                    "The app will still record and store your answers."
                ],
            }

        # Any other 4xx/5xx -> log and fall back
        if response.status_code >= 400:
            print(f"OpenRouter error status {response.status_code}: {response.text}")
            return {
                "score": 50,
                "shortFeedback": "AI service temporarily unavailable.",
                "detailedFeedback": "Please try again later.",
                "tips": [
                    "Check your network and try again.",
                    "Your responses are still valid practice."
                ],
            }

        data = response.json()
        if "choices" not in data or not data["choices"]:
            print("No choices in OpenRouter response:", data)
            return {
                "score": 50,
                "shortFeedback": "AI response missing.",
                "detailedFeedback": "Could not parse the AI's output.",
                "tips": ["Try again in a moment.", "Ensure your text is clear and complete."],
            }

        ai_text = data["choices"][0]["message"]["content"].strip()

        # Try to parse JSON
        try:
            parsed = json.loads(ai_text)
            score = min(max(0, int(parsed.get("score", 50))), 100)
            short = parsed.get("shortFeedback", "Good response.")
            detailed = parsed.get("detailedFeedback", "Your response is clear and on topic.")
            tips = parsed.get("tips", [])
            if not isinstance(tips, list):
                tips = []

            return {
                "score": score,
                "shortFeedback": short,
                "detailedFeedback": detailed,
                "tips": tips,
            }
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            print("Failed to parse OpenRouter JSON:", e)
            print("Raw AI text:", ai_text)

    except requests.RequestException as e:
        print("Request to OpenRouter failed:", e)
    except Exception as e:
        print("Unexpected error in get_llm_feedback:", e)

    # Final fallback
    return {
        "score": 50,
        "shortFeedback": "Feedback unavailable right now.",
        "detailedFeedback": "The AI service couldn't respond, but your answers are still valid practice.",
        "tips": [
            "Keep your responses specific and clear.",
            "Practice focusing on empathy or logical structure as relevant.",
        ],
    }


# === Expose /feedback ===

@app.post("/feedback", response_model=FeedbackResult)
def get_feedback(request: FeedbackRequest) -> FeedbackResult:
    raw = get_llm_feedback(
        category=request.category,
        prompt=request.prompt,
        user_response=request.userResponse,
    )

    return FeedbackResult(
        score=raw["score"],
        shortFeedback=raw["shortFeedback"],
        detailedFeedback=raw["detailedFeedback"],
        tips=raw["tips"],
    )