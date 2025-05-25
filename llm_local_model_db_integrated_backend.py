# run_agent_with_explanation.py
import os
import configparser
import logging
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain_predictor_tool import predict_and_explain_adherence_tool
from dotenv import load_dotenv
from reviq_helper import get_sqlite_tools
from langchain.schema import SystemMessage

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


config = configparser.ConfigParser()
config.read('config.ini')

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get the logger
logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4.1",
                 temperature=0.7,
                openai_api_key=openai_api_key
                 )

tools = [predict_and_explain_adherence_tool]

# 👇 SQLite DB Tool
sqlite_db_path = config["DEFAULT"]["sqlite_db_path"]

logger.info(f"sqlite_db_path : {sqlite_db_path}")

sql_tools = get_sqlite_tools(sqlite_db_path, llm)

# 👇 Combine tools
all_tools = tools + sql_tools

agent = initialize_agent(
    tools=all_tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs={
        "system_message": """You are a healthcare assistant. Only answer questions related to patient behavior, 
        medication adherence, and healthcare data. Reject any other topics."""
    }
)

if __name__ == "__main__":
    # result = agent.run(
    #     "Run a prediction for a 23-year-old male from Texas, zip code 77001, income grade 4, "
    #     "suffering from a chronic condition, with 4 dependents, working as a truck driver, married."
    # )
    result = agent.run(
        "Tell me a joke"
    )
    print(result)
