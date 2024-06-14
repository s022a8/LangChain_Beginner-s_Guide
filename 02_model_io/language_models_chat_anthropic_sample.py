# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage

chat = ChatOpenAI(
    model="gpt-3.5-turbo",
)

result = chat(
    [
        SystemMessage(content="あなたは親しい友人です。返答は敬語を使わず、フランクに会話してください。"),
        HumanMessage(content="こんにちは！"),
    ]
)

print(result.content)