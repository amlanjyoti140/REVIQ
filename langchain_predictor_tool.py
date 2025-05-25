# langchain_predictor_tool.py
from langchain.tools import tool
import pandas as pd
from behaviour_score_generator import calculate_adherance_score
from reviq_score_predictor import (
    predict_refill_reminder_score,
    predict_price_sensitivity_score,
    predict_awareness_score,
    predict_coverage_confusion_score
)

@tool
def predict_and_explain_adherence_tool(
    age: int,
    gender: str,
    state: str,
    city: str,
    zip_code: int,
    income_grade: int,
    condition: str,
    no_of_dependents: int,
    occupation: str,
    marital_status: str
) -> str:
    """
    Predict the patient's medication adherence score based on personal and socio-economic data.

    Args:
        age: Age of the patient (in years).
        gender: Gender of the patient (e.g., Male,Female).
        state: US state of residence (e.g., TX,CA).
        city: City of residence (e.g., Palo Alto, Omaha).
        zip_code: ZIP code of residence.
        income_grade: Income grade (1–4, where 4 is highest).
        condition: Chronic health condition (e.g., acute/chronic).
        no_of_dependents: Number of dependents.
        occupation: Job title or type (e.g., truck driver).
        marital_status: Marital status (e.g., Single, Married).

    Returns:
        A detailed explanation of the predicted adherence score and contributing factors.
    """
    patient = {
        "age": age,
        "gender": gender,
        "state": state,
        "zip_code": zip_code,
        "annual_income_grade": income_grade,
        "patient_condition": condition,
        "no_of_dependant": no_of_dependents,
        "occupation": occupation,
        "maritial_status": marital_status,
        "city": city,  # Optional dummy data if needed
        "id": 0, "name": "", "address_line1": "", "address_line2": "", "email": "", "phone": 0
    }

    df = pd.DataFrame([patient])
    df = predict_refill_reminder_score(df)
    df = predict_price_sensitivity_score(df)
    df = predict_awareness_score(df)
    df = predict_coverage_confusion_score(df)
    df = calculate_adherance_score(df)

    row = df.iloc[0]
    explanation = (
        f"Adherence score: {row['adherence_score']}\n"
        f"Refill reminder score: {row['refill_reminder_score']}\n"
        f"Price sensitivity score: {row['price_sensitivity_score']}\n"
        f"Awareness score: {row['awareness_score']}\n"
        f"Coverage confusion score: {row['coverage_confusion_score']}\n"
        f"This score reflects a {age}-year-old {gender} from {state}, with income grade {income_grade},\n"
        f"a {condition} condition, working as a {occupation}, {marital_status}, with {no_of_dependents} dependents."
    )

    return explanation
