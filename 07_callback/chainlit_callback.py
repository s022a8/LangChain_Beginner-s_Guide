from gc import callbacks
import chainlit as cl
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

chat = ChatOpenAI(
    temperature=0,
    # model="gpt-4"
    model="gpt-3.5-turbo"
)

tavily_search_tool = TavilySearchAPIWrapper()
tavily_search_results = TavilySearchResults(api_wrapper=tavily_search_tool, max_results=1)

tools = [tavily_search_results]

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


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Agentの初期化が完了しました。").send()

@cl.on_message
async def on_message(message: cl.Message):
    result = agent_executer.invoke(
        {"input": message.content},
        {"callbacks": [cl.LangchainCallbackHandler()]}
    )
    await cl.Message(content=result).send()
    