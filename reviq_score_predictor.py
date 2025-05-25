import h2o
import os
import configparser
import logging
import pandas as pd
from tabulate import tabulate
from behaviour_score_generator import calculate_adherance_score

# Create a ConfigParser object
config = configparser.ConfigParser()
config.read('config.ini')

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

model_saved_to_path = config["DEFAULT"]["model_saved_to_path"]
logger.info(f"model_saved_to_path: {model_saved_to_path}")

# Initialize H2O (do once at start)
h2o.init()

def _predict_score(patient_input, model_path: str, score_column_name: str) -> pd.DataFrame:
    """
    Predict scores using a model, and return input DataFrame with score_column_name appended.

    :param patient_input: pd.Series (single row) or pd.DataFrame (multiple rows)
    :param model_path: path to the saved H2O model
    :param score_column_name: name of the column to append with predictions
    :return: pd.DataFrame with prediction column added
    """
    categorical_cols = [
        'city', 'zip_code', 'state', 'gender', 'maritial_status',
        'occupation', 'patient_condition', 'annual_income_grade',
        'no_of_dependant'
    ]

    # Convert Series to single-row DataFrame
    if isinstance(patient_input, pd.Series):
        patient_df = pd.DataFrame([patient_input])
    elif isinstance(patient_input, pd.DataFrame):
        patient_df = patient_input.copy()
    else:
        raise TypeError("patient_input must be a pandas Series or DataFrame")

    model = h2o.load_model(model_path)
    patient_h2o = h2o.H2OFrame(patient_df)

    for col in categorical_cols:
        if col in patient_df.columns:
            patient_h2o[col] = patient_h2o[col].asfactor()

    features = model._model_json['output']['names'][:-1]
    logger.info(f"Using features: {features}")

    preds = model.predict(patient_h2o[features])
    patient_df[score_column_name] = preds.as_data_frame().iloc[:, 0].round(2)

    return patient_df



def predict_refill_reminder_score(patient_input) -> pd.DataFrame:
    model_path = os.path.join(model_saved_to_path, config["MODEL_NAMES"]["refill_reminder_score"])
    return _predict_score(patient_input, model_path, score_column_name="refill_reminder_score")

def predict_price_sensitivity_score(patient_input) -> pd.DataFrame:
    model_path = os.path.join(model_saved_to_path, config["MODEL_NAMES"]["price_sensitivity_score"])
    return _predict_score(patient_input, model_path, score_column_name="price_sensitivity_score")

def predict_awareness_score(patient_input) -> pd.DataFrame:
    model_path = os.path.join(model_saved_to_path, config["MODEL_NAMES"]["awareness_score"])
    return _predict_score(patient_input, model_path, score_column_name="awareness_score")

def predict_coverage_confusion_score(patient_input) -> pd.DataFrame:
    model_path = os.path.join(model_saved_to_path, config["MODEL_NAMES"]["coverage_confusion_score"])
    return _predict_score(patient_input, model_path, score_column_name="coverage_confusion_score")


def predict_all_scores(patient_input: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts all activity scores and calculates the adherence score.

    :param patient_input: Input patient data (single row or batch)
    :return: DataFrame with activity and adherence scores
    """
    df = predict_refill_reminder_score(patient_input)
    df = predict_price_sensitivity_score(df)
    df = predict_awareness_score(df)
    df = predict_coverage_confusion_score(df)
    df = calculate_adherance_score(df)
    return df


if __name__ == "__main__":
    # Example new patient record
    new_patient_df = pd.DataFrame([{
        "id": 1,
        "name": "Alice",
        "address_line1": "456 Elm St",
        "address_line2": "",
        "city": "Metropolis",
        "state": "NY",
        "zip_code": 10001,
        "age": 35,
        "email": "alice@example.com",
        "phone": 1234567890,
        "gender": "Female",
        "maritial_status": "Single",
        "occupation": "Nurse",
        "annual_income_grade": 2,
        "patient_condition": "acute",
        "no_of_dependant": 5
    },
        {
        "id": 1,
        "name": "Alice",
        "address_line1": "456 Elm St",
        "address_line2": "",
        "city": "Gordonberg",
        "state": "CA",
        "zip_code": 98491,
        "age": 50,
        "email": "alice@example.com",
        "phone": 1234567890,
        "gender": "Male",
        "maritial_status": "Single",
        "occupation": "Teacher",
        "annual_income_grade": 4,
        "patient_condition": "chronic",
        "no_of_dependant": 2
        }
    ])


    # Predict scores
    model_dir = model_saved_to_path
    refill = predict_refill_reminder_score(new_patient_df)
    price = predict_price_sensitivity_score(new_patient_df)
    aware = predict_awareness_score(new_patient_df)
    confuse = predict_coverage_confusion_score(new_patient_df)
    all_score = predict_all_scores(new_patient_df)

    print(f"Refill Reminder Score: {tabulate(refill.head())}")
    print(f"Price Sensitivity Score: {tabulate(price.head())}")
    print(f"Awareness Score: {tabulate(aware.head())}")
    print(f"Coverage Confusion Score: {tabulate(confuse.head())}")
    print(f"all_score Score: {tabulate(all_score.head())}")



