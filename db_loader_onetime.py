import pandas as pd
import configparser
import logging
from reviq_helper import load_df_to_sqlite

# Create a ConfigParser object
config = configparser.ConfigParser()
config.read('config.ini')

# ********************************************  Getting the config parameters ****************************************

src_dir = config["DEFAULT"]["input_file_dir"]
input_patient_file_nm = config["DEFAULT"]["patient_file_name"]
input_activity_log_file_nm = config["DEFAULT"]["activity_log_file_name"]
input_income_range_file_nm = config["DEFAULT"]["income_range_file_name"]
sqlite_db_path = config["DEFAULT"]["sqlite_db_path"]

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get the logger
logger = logging.getLogger(__name__)

# ********************************************  Loader functions ****************************************

def patient_dtl_loader() -> None:
    logger.info(f"src_dir : {src_dir}")
    logger.info(f"input_patient_file_nm : {input_patient_file_nm}")
    logger.info(f"sqlite_db_path : {sqlite_db_path}")

    df_patient = pd.read_csv(f"{src_dir}/{input_patient_file_nm}")

    load_df_to_sqlite(df=df_patient,
                      table_name="patient_dtl",
                      sqlite_db_path=sqlite_db_path)

    logger.info(f"db load to patient_dtl done")


def activity_log_loader() -> None:
    logger.info(f"src_dir : {src_dir}")
    logger.info(f"input_activity_log_file_nm : {input_activity_log_file_nm}")
    logger.info(f"sqlite_db_path : {sqlite_db_path}")

    df_activity_log = pd.read_csv(f"{src_dir}/{input_activity_log_file_nm}")

    load_df_to_sqlite(df=df_activity_log,
                      table_name="activity_log",
                      sqlite_db_path=sqlite_db_path)

    logger.info(f"db load to patient_dtl done")


def income_range_loader() -> None:

    logger.info(f"src_dir : {src_dir}")
    logger.info(f"input_income_range_file_nm : {input_income_range_file_nm}")
    logger.info(f"sqlite_db_path : {sqlite_db_path}")

    df_income_range = pd.read_csv(f"{src_dir}/{input_income_range_file_nm}")

    load_df_to_sqlite(df=df_income_range,
                      table_name="income_range_grade",
                      sqlite_db_path=sqlite_db_path)

    logger.info(f"db load to income_range_grade done")


if __name__ == '__main__':
    logger.info("  Loading started...")

    # ********************************************  Calling the loaders ************************************************

    patient_dtl_loader()

    activity_log_loader()

    income_range_loader()

    logger.info("  Loading complete...")
