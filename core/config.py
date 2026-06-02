import os
from dotenv import load_dotenv

# Load environment variables once when the app starts
load_dotenv()

SYSTEM_PROMPT = """You are a strict binary classifier. 
Analyze the provided tabular data row and predict the label.
You MUST output ONLY the digit 0 or 1.
Do not include any formatting, markdown, explanations, or extra text."""

MOCK_DATASET = [
    {"age": 45, "education": "Bachelors", "hours_per_week": 40},
    {"age": 22, "education": "High School", "hours_per_week": 20},
    {"age": 55, "education": "Doctorate", "hours_per_week": 50},
    {"age": 30, "education": "Some College", "hours_per_week": 35},
    {"age": 38, "education": "Masters", "hours_per_week": 45},
]

COT_SYSTEM_PROMPT = """You are a meticulous data scientist.
Analyze the provided tabular data row and predict the label (0 or 1).
Think step-by-step about the features and how they might relate to the target class.
Explain your reasoning briefly but clearly.
You MUST conclude your response with the exact phrase "Final Answer: " followed by either 0 or 1."""
