# run_agent_with_explanation.py
import os
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain_predictor_tool import predict_and_explain_adherence_tool
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(model="gpt-4.1",
                 temperature=0.7,
                openai_api_key=openai_api_key)

tools = [predict_and_explain_adherence_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=False
)

if __name__ == "__main__":
    result = agent.run(
        "Run a prediction for a 23-year-old male from Texas, zip code 77001, income grade 4, "
        "suffering from a chronic condition, with 4 dependents, working as a truck driver, married."
    )
    print(result)
