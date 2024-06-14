from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

from langchain.schema.output_parser import StrOutputParser
from langchain.callbacks.tracers import ConsoleCallbackHandler

prompt = PromptTemplate(
    template="{product}はどこの会社が開発した製品ですか？",
    input_variables=[
        "product"
    ]
)
chat = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

chain: RunnableSequence = prompt | chat | output_parser

result = chain.invoke({"product": "iPhone"}, config={'callbacks': [ConsoleCallbackHandler()]})

print(result)