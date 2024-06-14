# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.schema import AIMessage

chat = ChatOpenAI(
    model="gpt-3.5-turbo",
)

result = chat(
    [
        HumanMessage(content="こんにちは！"),
        AIMessage(content="{ChatModelからの返答である茶碗蒸しの作り方}"),
        HumanMessage(content="英語に翻訳して"),
    ]
)

print(result.content)