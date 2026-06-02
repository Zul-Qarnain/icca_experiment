import os
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
from core.config import SYSTEM_PROMPT

class APIError(Exception): pass

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(2))
def call_openrouter(prompt: str, model_name: str = "meta-llama/llama-3.2-3b-instruct:free") -> str:
    try:
        response = openrouter_client.chat.completions.create(
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
        raise APIError(f"OpenRouter API failed: {e}")
