# ==========================================
# DYNAMIC PROMPT DICTIONARIES BY DATASET
# ==========================================

ZERO_SHOT = {
    "adult.csv": "You are a binary classifier. Predict income class (0 for <=50K, 1 for >50K). Output ONLY 0 or 1.",
    "german_credit.csv": "You are a binary classifier. Predict credit risk (0 for Good Credit, 1 for Bad Credit). Output ONLY 0 or 1.",
    "Heart_disease_cleveland_new.csv": "You are a binary classifier. Predict presence of heart disease (0 for absence, 1 for presence). Output ONLY 0 or 1."
}

FEW_SHOT_STANDARD = {
    "adult.csv": """You are a binary classifier. Predict income class (0 for <=50K, 1 for >50K). Output ONLY 0 or 1.
Examples:
Input: Age: 25, Workclass: Private, Education: 11th, Capital-Gain: 0 -> Label: 0
Input: Age: 44, Workclass: Private, Education: Bachelors, Capital-Gain: 7688 -> Label: 1
Input: Age: 34, Workclass: Private, Education: 10th, Capital-Gain: 0 -> Label: 0
Now classify this row:""",

    "german_credit.csv": """You are a binary classifier. Predict credit risk (0 for Good Credit, 1 for Bad Credit). Output ONLY 0 or 1.
Examples:
Input: checking_status: A14, duration: 12, credit_amount: 2096, personal_status: A93 -> Label: 0
Input: checking_status: A11, duration: 48, credit_amount: 5951, personal_status: A92 -> Label: 1
Input: checking_status: A12, duration: 24, credit_amount: 2028, personal_status: A93 -> Label: 0
Now classify this row:""",

    "Heart_disease_cleveland_new.csv": """You are a binary classifier. Predict presence of heart disease (0 for absence, 1 for presence). Output ONLY 0 or 1.
Examples:
Input: age: 63, sex: 1, cp: 0, chol: 233, thalach: 150 -> Label: 0
Input: age: 67, sex: 1, cp: 3, chol: 286, thalach: 108 -> Label: 1
Input: age: 41, sex: 0, cp: 1, chol: 204, thalach: 172 -> Label: 0
Now classify this row:"""
}

FEW_SHOT_COT = {
    "adult.csv": """You are a binary classifier. Predict income class (0 for <=50K, 1 for >50K).
Examples:
Input: Age: 25, Education: 11th -> Output: <thinking>Low education and young age strongly correlate with low income.</thinking><answer>0</answer>
Input: Age: 44, Education: Bachelors, Capital-Gain: 7688 -> Output: <thinking>High capital gains and a degree indicate higher wealth brackets.</thinking><answer>1</answer>
Analyze step-by-step inside <thinking> tags, then output ONLY 0 or 1 inside <answer> tags. Classify this row:""",

    "german_credit.csv": """You are a binary classifier. Predict credit risk (0 for Good Credit, 1 for Bad Credit).
Examples:
Input: duration: 12, credit_history: A34 -> Output: <thinking>Short duration and good history implies reliable payback.</thinking><answer>0</answer>
Input: duration: 48, checking_status: A11 -> Output: <thinking>Long duration with poor checking balance indicates high default risk.</thinking><answer>1</answer>
Analyze step-by-step inside <thinking> tags, then output ONLY 0 or 1 inside <answer> tags. Classify this row:""",

    "Heart_disease_cleveland_new.csv": """You are a binary classifier. Predict presence of heart disease (0 for absence, 1 for presence).
Examples:
Input: cp: 0, thalach: 150 -> Output: <thinking>No chest pain and healthy heart rate suggests no disease.</thinking><answer>0</answer>
Input: cp: 3, chol: 286, oldpeak: 1.5 -> Output: <thinking>High cholesterol, severe chest pain, and ST depression are strong indicators of disease.</thinking><answer>1</answer>
Analyze step-by-step inside <thinking> tags, then output ONLY 0 or 1 inside <answer> tags. Classify this row:"""
}
