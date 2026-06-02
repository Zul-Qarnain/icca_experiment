import os
from groq import Groq
from tenacity import retry, wait_exponential, stop_after_attempt
from core.config import SYSTEM_PROMPT

class APIError(Exception): pass

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(2))
def call_groq(prompt: str, model_name: str = "llama-3.1-8b-instant") -> str:
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model=model_name,
            temperature=0.0,
            max_tokens=5,
            timeout=15.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise APIError(f"Groq API failed: {e}")
