from langchain.chains.retrieval_qa.base import RetrievalQA

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo")

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

database = Chroma(
    persist_directory="./.data",
    embedding_function=embeddings
)

retriever = database.as_retriever()

qa = RetrievalQA.from_llm(
    llm=chat,
    retriever=retriever,
    return_source_documents=True
)

result = qa.invoke("飛行者の最高速度は？")

print(result["result"])
print(result["source_documents"])