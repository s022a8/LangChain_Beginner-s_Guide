from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI(
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler()
    ]
)

resp = chat.invoke([
    HumanMessage(content="美味しいステーキの焼き方を教えて")
])