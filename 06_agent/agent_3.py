import random
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_community.tools.file_management.write import WriteFileTool
from langchain.agents import Tool

chat = ChatOpenAI(
    temperature=0,
    # model="gpt-4"
    model="gpt-3.5-turbo"
)

tools = []

write_file_tool = WriteFileTool(root_dir="./")

def min_limit_random_number(min_number):
    return random.randint(int(min_number), 100000)

tools.append(write_file_tool)

tools.append(
    Tool(
        name="Random",
        description="特定の最小値以上のランダムな数字を生成することができます。",
        func=min_limit_random_number
    )
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_openai_tools_agent(llm=chat, tools=tools, prompt=prompt)
agent_executer = AgentExecutor(agent=agent, tools=tools, verbose=True)
# agent_executer = AgentExecutor(agent=agent, tools=tools)

result = agent_executer.invoke({"input": "10以上のランダムな数字を生成してrandom.txtというファイルに保存してください。"})

print(f"実行結果: {result}")