import os
import google.generativeai as genai
from tenacity import retry, wait_exponential, stop_after_attempt
from core.config import SYSTEM_PROMPT

class APIError(Exception): pass

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel(
    "gemini-2.0-flash",
    system_instruction=SYSTEM_PROMPT
)

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def call_gemini(prompt: str) -> str:
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
        return response.text
    except Exception as e:
        raise APIError(f"Gemini API failed: {e}")
