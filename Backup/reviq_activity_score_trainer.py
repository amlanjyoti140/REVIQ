import h2o
import configparser
import logging
import os
import pandas as pd
from tabulate import tabulate
from h2o.automl import H2OAutoML
from reviq_helper import read_table_from_sqlite

# Create a ConfigParser object
config = configparser.ConfigParser()
config.read('config.ini')

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

sqlite_db_path = config["DEFAULT"]["sqlite_db_path"]
model_saved_to_path = config["DEFAULT"]["model_saved_to_path"]


logger.info(f"sqlite_db_path : {sqlite_db_path}")
logger.info(f"model_saved_to_path : {model_saved_to_path}")


def train_activity_score_models(df: pd.DataFrame, target_columns, save_dir="h2o_models"):
    """
    Train and save separate H2O AutoML models_BKP for each activity score.

    :param df: pandas DataFrame with features and target scores
    :param target_columns: list of activity score names
    :param save_dir: directory to save trained models_BKP
    :return: dict of {target_column: model_path}
    """
    h2o.init(max_mem_size_GB=4)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_h2o = h2o.H2OFrame(df)

    # Explicitly set categorical columns
    categorical_cols = [
        'city',
        'zip_code',
        'state',
        'gender',
        'maritial_status',
        'occupation',
        'patient_condition',
        'annual_income_grade',  # treated as category, not continuous
        'no_of_dependant'  # small range, also safe to treat as categorical
    ]

    for col in categorical_cols:
        if col in df.columns:
            df_h2o[col] = df_h2o[col].asfactor()

    ignore_base = ['id', 'name', 'phone', 'email', 'address_line1', 'address_line2','adherence_score']
    saved_models = {}

    for target in target_columns:
        logger.info(f"Training model to predict: {target}")
        columns_to_ignore = ignore_base + [col for col in target_columns if col != target]
        features = [col for col in df.columns if col not in columns_to_ignore + [target]]

        logger.info(f"Considering features.....{features}")

        aml = H2OAutoML(max_models=10, seed=42)
        aml.train(x=features, y=target, training_frame=df_h2o)

        best_model = aml.leader
        # model_path = h2o.save_model(model=best_model, path=save_dir, force=True)
        lb = aml.leaderboard
        logger.info(f"Showing leaderboard : {lb}")

        algo_name = best_model.algo if hasattr(best_model, "algo") else "unknown"
        # model_name = f"{target}_{algo_name}_model"
        logger.info(f"Best algorithm chosen : {algo_name}")
        model_name = config["MODEL_NAMES"][target]
        logger.info(f"Training model name in config: {model_name}")

        model_path = h2o.save_model(model=best_model, path=save_dir, filename=model_name, force=True)

        logger.info(f"Saved best model for {target} at: {model_path}")
        saved_models[target] = model_path

    return saved_models


if __name__ == "__main__":

    df_patient = read_table_from_sqlite(sqlite_db_path=sqlite_db_path, table_name="patient_matrix")

    logger.info(df_patient.columns)
    logger.info(df_patient.dtypes)
    training_targets = ["refill_reminder_score", "price_sensitivity_score", "awareness_score",
                        "coverage_confusion_score"]

    models = train_activity_score_models(df=df_patient,
                                         target_columns=training_targets,
                                         save_dir=model_saved_to_path)
