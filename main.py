import configparser
import logging
from raw_data_reader import patient_dtl_reader
from reviq_helper import load_df_to_sqlite
from behaviour_score_generator import score_generator



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # **************************************** config setup *********************************************

    # Create a ConfigParser object
    config = configparser.ConfigParser()
    config.read('config.ini')


    src_dir = config["DEFAULT"]["input_file_dir"]
    input_patient_file_nm = config["DEFAULT"]["patient_file_name"]
    sqlite_db_path = config["DEFAULT"]["sqlite_db_path"]


    # Configure the logger
    logging.basicConfig(
        level=logging.INFO,  # Set the minimum logging level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Get the logger
    logger = logging.getLogger(__name__)

    logger.info(f"src_dir : {src_dir}")
    logger.info(f"input_patient_file_nm : {input_patient_file_nm}")
    logger.info(f"sqlite_db_path : {sqlite_db_path}")

    patient_df = patient_dtl_reader(src_file_with_path=f"{src_dir}/{input_patient_file_nm}")

    logger.info(patient_df)

    score_generator





    # load_df_to_sqlite(df=patient_df,
    #                   table_name="patient_dtl",
    #                   sqlite_db_path=sqlite_db_path)


