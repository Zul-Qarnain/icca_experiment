import os
import time
import requests
from dotenv import load_dotenv
from core.parser import extract_label

load_dotenv()

# Load Keys
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")

def make_request_with_retry(url, headers, payload, timeout=30):
    """Exponential backoff to handle free-tier Rate Limits (HTTP 429)."""
    for attempt in range(7):
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if res.status_code == 200:
                return res.json()
            elif res.status_code == 429:
                time.sleep(2 ** attempt) # Sleep 1, 2, 4, 8, 16 seconds
                continue
            else:
                print(f"  [API Error] {res.status_code} - {res.text}")
                return None
        except Exception as e:
            print(f"  [Network Exception] {str(e)}")
            time.sleep(2 ** attempt)
    return None

def call_google_studio(prompt_sys, prompt_user, mode='standard'):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_KEY}"
    payload = {
        "contents": [{"parts": [{"text": f"{prompt_sys}\n\nInput: {prompt_user}"}]}],
        "generationConfig": {"temperature": 0.1}
    }
    res = make_request_with_retry(url, {"Content-Type": "application/json"}, payload)
    if res:
        text = res["candidates"][0]["content"]["parts"][0]["text"]
        return text, extract_label(text, mode)
    return "Error", -2

def call_groq(prompt_sys, prompt_user, mode='standard'):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "system", "content": prompt_sys}, {"role": "user", "content": prompt_user}],
        "temperature": 0.1
    }
    res = make_request_with_retry(url, headers, payload)
    if res:
        text = res["choices"][0]["message"]["content"]
        return text, extract_label(text, mode)
    return "Error", -2

def call_openrouter(prompt_sys, prompt_user, mode='standard'):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "messages": [{"role": "system", "content": prompt_sys}, {"role": "user", "content": prompt_user}],
        "temperature": 0.1
    }
    res = make_request_with_retry(url, headers, payload)
    if res:
        text = res["choices"][0]["message"]["content"]
        return text, extract_label(text, mode)
    return "Error", -2

def call_mistral(prompt_sys, prompt_user, mode='standard'):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "open-mistral-7b",
        "messages": [{"role": "system", "content": prompt_sys}, {"role": "user", "content": prompt_user}],
        "temperature": 0.1
    }
    res = make_request_with_retry(url, headers, payload)
    if res:
        text = res["choices"][0]["message"]["content"]
        return text, extract_label(text, mode)
    return "Error", -2

def call_nvidia_nim(prompt_sys, prompt_user, enable_thinking=True, mode='standard'):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {NVIDIA_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [{"role": "system", "content": prompt_sys}, {"role": "user", "content": prompt_user}],
        "temperature": 0.1,
        "max_tokens": 4096,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            "reasoning_budget": 16384 if enable_thinking else 0
        }
    }
    res = make_request_with_retry(url, headers, payload, timeout=45)
    if res:
        text = res["choices"][0]["message"]["content"]
        return text, extract_label(text, mode)
    return "Error", -2
