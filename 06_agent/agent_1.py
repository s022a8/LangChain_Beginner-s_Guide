from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_community.tools.requests.tool import RequestsGetTool
from langchain_community.utilities.requests import TextRequestsWrapper

chat = ChatOpenAI(
    temperature=0,
    model="gpt-4"
    # model="gpt-3.5-turbo"
)

requests_wrapper = TextRequestsWrapper()
request_tool = RequestsGetTool(requests_wrapper=requests_wrapper, allow_dangerous_requests=True)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "以下のURLにアクセスして東京の天気を調べて日本語で答えてください。\n{url}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

tools = [request_tool]

agent = create_openai_tools_agent(llm=chat, tools=tools, prompt=prompt)
agent_executer = AgentExecutor(agent=agent, tools=tools, verbose=True)
# agent_executer = AgentExecutor(agent=agent, tools=tools)

result = agent_executer.invoke({"url": "https://www.jma.go.jp/bosai/forecast/data/overview_forecast/130000.json"})

print(f"実行結果: {result}")