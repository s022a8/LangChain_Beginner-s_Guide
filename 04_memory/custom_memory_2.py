import os
import chainlit as cl
from pydantic.v1 import Field, validator
from typing import Dict, Any
import redis

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains.conversation.base import ConversationChain

from langchain.schema import HumanMessage, AIMessage

class RedisMemory(ConversationSummaryMemory):
    session_id: str
    redis_attr: redis.Redis = None
    
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
        # Redisに保存
        self.redis_attr.set(self.session_id, self.buffer)

    def load_memory_variables(self, inputs):
        """Redis からチャット履歴を読み込み"""
        # Redisから履歴を取得
        buffer_string = self.redis_attr.get(self.session_id)
        # 既存の履歴をクリア
        self.chat_memory.clear()
        # 履歴をメモリにロード
        if buffer_string:    
            messages = buffer_string.decode("utf-8").split("\n")
            for i in range(0, len(messages), 1):
                if (i % 2 == 0):
                    tmp_human_message = messages[i].lstrip('Human: *')
                    human_message = HumanMessage(content=tmp_human_message)
                    self.chat_memory.add_user_message(human_message)
                else:
                    tmp_ai_message = messages[i].lstrip('AI: *')
                    ai_message = AIMessage(content=tmp_ai_message)
                    self.chat_memory.add_ai_message(ai_message)
        # 履歴を辞書形式で返す
        return {"history": self.chat_memory.messages}

chat = ChatOpenAI(model="gpt-3.5-turbo")


@cl.on_chat_start
async def on_chat_start():
    thread_id = None
    while not thread_id:
        res = await cl.AskUserMessage(content="私は会話の文脈を考慮した返答ができるチャットボットです。\
    スレッドIDを入力してください。", timeout=600).send()
        if res:
            thread_id = res['output']
    
    chain = ConversationChain(
        memory=RedisMemory(
            session_id=thread_id,
            llm=chat
        ),
        llm=chat
    )
    
    memory_message_result = chain.memory.load_memory_variables({})
    messages = memory_message_result['history']
    
    for message in messages:
        if isinstance(message, HumanMessage):
            await cl.Message(
                author="User",
                content=f"{message.content}",
            ).send()
        else:
             await cl.Message(
                author="ChatBot",
                content=f"{message.content}",
            ).send()
    
    cl.user_session.chain = chain

@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.chain
    result = chain.invoke(message.content)
    await cl.Message(content=result["response"]).send()