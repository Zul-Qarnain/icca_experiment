import os
from mistralai.client import Mistral
from tenacity import retry, wait_exponential, stop_after_attempt
from core.config import SYSTEM_PROMPT

class APIError(Exception): pass

mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"), timeout_ms=15000)

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(2))
def call_mistral_direct(prompt: str, model_name: str = "mistral-small-latest") -> str:
    try:
        response = mistral_client.chat.complete(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=5,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise APIError(f"Official Mistral API failed: {e}")
