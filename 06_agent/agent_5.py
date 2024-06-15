import os
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.retrievers import WikipediaRetriever
from langchain.chains.conversation.base import ConversationBufferMemory
from pydantic.v1 import Field, validator
from typing import Dict, Any
from langchain.schema import HumanMessage, AIMessage
import redis
import json
from bs4 import BeautifulSoup

class RedisMemory(ConversationBufferMemory):
    session_id: str
    redis_attr: redis.Redis = None
    return_messages: bool
    
    @validator("redis_attr", always=True)
    def validator_redis_attr(cls, field, values: Dict[str, Any]):
        try:
            field = redis.Redis(
                host=os.environ['REDIS_HOST_1'],
                port=os.environ['REDIS_PORT'],
                password=os.environ['REDIS_PASSWORD_1'],
                ssl=True)
        except redis.exceptions.ConnectionError as e:
            print("Redis connection error:", e)
        return field

    def save_context(self, inputs, outputs):
        """Redis にチャット履歴を保存"""
        # メモリに履歴を追加
        super().save_context(inputs, outputs)
        # 履歴をJSON形式にシリアライズしてRedisに保存
        serialized_buffer = json.dumps([self.message_to_dict(message) for message in self.buffer])
        self.redis_attr.set(self.session_id, serialized_buffer)

    def load_memory_variables(self, inputs):
        """Redis からチャット履歴を読み込み"""
        # Redisから履歴を取得
        buffer_string = self.redis_attr.get(self.session_id)
        # 既存の履歴をクリア
        self.chat_memory.clear()
        # 履歴をメモリにロード
        if buffer_string:    
            # JSON形式をデシリアライズ
            messages = json.loads(buffer_string.decode("utf-8"))
            for message in messages:
                if message['type'] == 'human':
                    human_message = HumanMessage(content=message['content'])
                    self.chat_memory.add_user_message(human_message)
                elif message['type'] == 'ai':
                    ai_message = AIMessage(content=message['content'])
                    self.chat_memory.add_ai_message(ai_message)
        # 履歴を辞書形式で返す
        return {"history": self.chat_memory.messages}

    @staticmethod
    def message_to_dict(message):
        """メッセージオブジェクトを辞書形式に変換"""
        if isinstance(message, HumanMessage):
            return {"type": "human", "content": message.content}
        elif isinstance(message, AIMessage):
            return {"type": "ai", "content": message.content}
        else:
            raise TypeError(f"Unsupported message type: {type(message)}")

# BeautifulSoupを使った解析関数
def parse_wikipedia_content(html):
    soup = BeautifulSoup(html, features="html.parser")
    paragraphs = soup.find_all('p')
    content = "\n".join([p.get_text() for p in paragraphs])
    return content

chat = ChatOpenAI(
    temperature=0,
    # model="gpt-4"
    model="gpt-3.5-turbo"
)

tools = []

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

redisMemory = RedisMemory(
    session_id="chat_history_2",
    return_messages=True
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful assistant. 
         You will keep track of the conversation history and use it to answer follow-up questions.
         When asked to repeat a previous command, you should find the last command in the conversation history and repeat it.
         """),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_openai_tools_agent(llm=chat, tools=tools, prompt=prompt)
agent_executer = AgentExecutor(agent=agent, tools=tools, memory=redisMemory, verbose=True)
# agent_executer = AgentExecutor(agent=agent, tools=tools, memory=redisMemory)

# # 実行前のメモリ内容を確認
# print("実行前のメモリ内容:", redisMemory.load_memory_variables({}))
input_1 = "スコッチウィスキーについてWikipediaで調べて概要を日本語で概要をまとめてください。"
result = agent_executer.invoke({"input": input_1})
print(f"1回目の実行結果: {result}")

# # 実行後のメモリ内容を確認
# print("1回目実行後のメモリ内容:", redisMemory.load_memory_variables({}))
input_2 = "以前の指示をもう一度実行してください。"
last_human_message = next((m.content for m in redisMemory.chat_memory.messages[::-1] if isinstance(m, HumanMessage)), None)
input_2 = last_human_message if last_human_message else input_2

result_2 = agent_executer.invoke({"input": input_2})
print(f"2回目の実行結果: {result_2}")

# # 2回目実行後のメモリ内容を確認
# print("2回目実行後のメモリ内容:", redisMemory.load_memory_variables({}))