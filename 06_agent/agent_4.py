from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_community.tools.file_management.write import WriteFileTool
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.retrievers import WikipediaRetriever

chat = ChatOpenAI(
    temperature=0,
    # model="gpt-4"
    model="gpt-3.5-turbo"
)

tools = []

write_file_tool = WriteFileTool(root_dir="./")
tools.append(write_file_tool)

retriever = WikipediaRetriever(
    lang="ja",
    doc_content_chars_max=500,
    top_k_results=1,
)
tools.append(
    create_retriever_tool(
        name="WikipediaRetriever",
        description="受け取った単語に関するWikipediaの記事を取得できる",
        retriever=retriever,
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

result = agent_executer.invoke({"input": "スコッチウィスキーについてWikipediaで調べて概要を日本語でwhiskey.txtというファイルに保存してください。"})

print(f"実行結果: {result}")