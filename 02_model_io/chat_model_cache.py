import time
import langchain
from langchain_community.cache import InMemoryCache
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

cache = InMemoryCache()

chat = ChatOpenAI(cache=cache)
start = time.time()
result = chat.invoke([
    HumanMessage(content="こんにちは！")
])

end = time.time()
print(result.content)
print(f"実行時間: {end - start}秒")

result = chat.invoke([
    HumanMessage(content="こんにちは！")
])
end2 = time.time()
print(result.content)
print(f"実行時間: {end2 - end}秒")