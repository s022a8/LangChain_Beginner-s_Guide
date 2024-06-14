from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.file_management.write import WriteFileTool

chat = ChatOpenAI(
    temperature=0,
    model="gpt-4"
    # model="gpt-3.5-turbo"
)

tavily_search_tool = TavilySearchAPIWrapper()
tavily_search_results = TavilySearchResults(api_wrapper=tavily_search_tool, max_results=1)

write_file_tool = WriteFileTool(root_dir="./")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

tools = [tavily_search_results, write_file_tool]

agent = create_openai_tools_agent(llm=chat, tools=tools, prompt=prompt)
agent_executer = AgentExecutor(agent=agent, tools=tools, verbose=True)
# agent_executer = AgentExecutor(agent=agent, tools=tools)

result = agent_executer.invoke({"input": "北海道の名産品を調べて日本語でresult.txtというファイルに保存してください。"})

print(f"実行結果: {result}")