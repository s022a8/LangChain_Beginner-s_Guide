import chainlit as cl

from langchain_openai import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain

from langchain.schema import HumanMessage

chat = ChatOpenAI(model="gpt-3.5-turbo")
memory = ConversationBufferMemory(return_messages=True)

chain = ConversationChain(
    memory=memory,
    llm=chat
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="私は会話の文脈を考慮した返答ができるチャットボットです。\
    メッセージを入力してください。").send()

@cl.on_message
async def on_message(message: cl.Message):
    result = chain.invoke(message.content)
    await cl.Message(content=result["response"]).send()