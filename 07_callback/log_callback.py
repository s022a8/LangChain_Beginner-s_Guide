import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage

class LogCallbackHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("ChatModelの実行を開始します....")
        print(f"入力: {messages}")
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        print("ChatModelの実行を開始します....")
        print(f"入力: {inputs}")

chat = ChatOpenAI(
    # model="gpt-4"
    model="gpt-3.5-turbo",
    callbacks=[
        LogCallbackHandler()
    ]
)

result = chat.invoke([
    HumanMessage(content="こんにちは！：")
])

print(result)