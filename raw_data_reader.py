import pandas as pd


def patient_dtl_reader(src_file_with_path:str)->pd.DataFrame :

    # This function read csv file and return a pd dataframe
    return pd.read_csv(src_file_with_path)



