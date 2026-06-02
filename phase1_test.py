from models.wrappers import call_google_studio, call_groq, call_openrouter, call_mistral, call_nvidia_nim

print("--- Testing API Connections ---")
test_input = "Age: 39, Workclass: Private, Education: Bachelors, Capital-Gain: 2174"

print("Google (Gemma 2):", call_google_studio("Predict 0 or 1", test_input))
print("Groq (Llama 3.1):", call_groq("Predict 0 or 1", test_input))
print("OpenRouter (Qwen 3):", call_openrouter("Predict 0 or 1", test_input))
print("Mistral (7B):", call_mistral("Predict 0 or 1", test_input))
print("NVIDIA (Nemotron Reasoning):", call_nvidia_nim("Predict 0 or 1", test_input))
print("--- All tests finished ---")
