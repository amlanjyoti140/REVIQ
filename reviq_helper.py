import pandas as pd
import sqlite3
from typing import Union
import logging
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import Tool
import os

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_df_to_sqlite(
    df: pd.DataFrame,
    table_name: str,
    sqlite_db_path: str,
    if_exists: str = 'replace'  # Options: 'fail', 'replace', 'append'
) -> None:
    """
    Load a DataFrame into a SQLite table.

    Parameters:
        df (pd.DataFrame): The DataFrame to load.
        table_name (str): The name of the table in the SQLite database.
        sqlite_db_path (str): Path to the SQLite database file.
        if_exists (str): What to do if the table already exists. Options:
                         'fail', 'replace', 'append'. Default is 'replace'.
    """
    logger.info(f"Connecting to SQLite DB at: {sqlite_db_path}")
    with sqlite3.connect(sqlite_db_path) as conn:
        logger.info(f"Loading DataFrame into table: {table_name}")
        df.to_sql(name=table_name, con=conn, if_exists=if_exists, index=False)
        logger.info("DataFrame successfully loaded.")
    conn.close()


def read_table_from_sqlite(sqlite_db_path: str, table_name: str) -> pd.DataFrame:
    """
    Reads a table from a SQLite database and returns it as a pandas DataFrame.

    Args:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to read.

    Returns:
        pd.DataFrame: DataFrame containing the table's contents.
    """
    logger.info(f"Connecting to SQLite DB at: {sqlite_db_path}")
    # Connect to the SQLite database
    conn = sqlite3.connect(sqlite_db_path)

    logger.info(f"Connecting to SQLite DB at: {sqlite_db_path} Done")
    # Query the table
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    # Close the connection
    conn.close()

    return df



def get_sqlite_tools(db_path: str, llm) -> list:

    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return toolkit.get_tools()

