import configparser
import logging
import pandas as pd
from tabulate import tabulate
from behaviour_score_generator import score_generator, calculate_adherance_score
from reviq_helper import read_table_from_sqlite, load_df_to_sqlite

# Create a ConfigParser object
config = configparser.ConfigParser()
config.read('config.ini')

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get the logger
logger = logging.getLogger(__name__)

src_dir = config["DEFAULT"]["input_file_dir"]
input_patient_file_nm = config["DEFAULT"]["patient_file_name"]
input_activity_log_file_nm = config["DEFAULT"]["activity_log_file_name"]
sqlite_db_path = config["DEFAULT"]["sqlite_db_path"]



df_patient = read_table_from_sqlite(sqlite_db_path=sqlite_db_path,
                                    table_name="patient_dtl")

df_activity_log = read_table_from_sqlite(sqlite_db_path=sqlite_db_path,
                                         table_name="activity_log")

df_activity_log['time_stamp'] = pd.to_datetime(df_activity_log['time_stamp'], errors='coerce')

df_income_range = read_table_from_sqlite(sqlite_db_path=sqlite_db_path,
                                         table_name="income_range_grade")

print(tabulate(df_patient.head(), headers='keys', tablefmt='psql'))
print(tabulate(df_activity_log.head(), headers='keys', tablefmt='psql'))
print(tabulate(df_income_range.head(), headers='keys', tablefmt='psql'))

df_patient_with_activity_score = score_generator(patients_df=df_patient,
                                                 activity_df=df_activity_log,
                                                 income_df=df_income_range)

print(tabulate(df_patient_with_activity_score.head(), headers='keys', tablefmt='psql'))


# **************************************** calculating adherance score **************************************

df_patient_with_all_score = calculate_adherance_score(df_patient_with_activity_score)

# df_patient_with_all_score.to_csv("/Users/amlanjyotipatnaik/PycharmProjects/REVIQ/Output/patient_with_all_score.csv")

logger.info("Loading df_patient_with_all_score to patient_matrix table..")

load_df_to_sqlite(df=df_patient_with_all_score,
                  table_name="patient_matrix",
                  sqlite_db_path=sqlite_db_path)

logger.info(f"Loading df_patient_with_all_score to patient_matrix table {df_patient_with_all_score.count()}..DONE")
