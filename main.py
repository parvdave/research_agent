from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from tools import search_tool, wiki_tool, save_to_txt

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


llm = ChatOpenAI(model="gpt-4o-mini")

tools = [search_tool, wiki_tool, save_to_txt]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a research assistant. "
        "Research the user's topic, use tools when necessary, "
        "and return a structured response."
    ),
    response_format=ResearchResponse,
)

query = input("What can I help you research? ")

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": query}
        ]
    }
)

print("Structured response:")
print(result["structured_response"])