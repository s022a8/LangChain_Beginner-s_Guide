from langchain_openai import OpenAI
from langchain_community.retrievers import WikipediaRetriever
from langchain.retrievers.re_phraser import RePhraseQueryRetriever

# from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables.base import RunnableSequence

retriever = WikipediaRetriever(
    lang="ja",
    doc_content_chars_max=500,
)

# llm_chain = LLMChain(
#     llm=ChatOpenAI(
#         temperature=0
#     ),
#     prompt=PromptTemplate(
#         input_variables=["question"],
#         template="""
#         以下の質問からWikipediaで検索するキーワードを抽出してください。
#         質問: {question}
#         """
#     )
# )

llm = OpenAI(temperature=0)

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    以下の質問からWikipediaで検索するキーワードを抽出してください。
    質問: {question}
    """
)

re_pharese_query_retriever = RePhraseQueryRetriever.from_llm(
    llm=llm,
    retriever=retriever,
    prompt=prompt
)

documents = re_pharese_query_retriever.invoke(
    "私はラーメンが好きです。ところでバーボンウィスキーとはなんですか？"
)

print(documents)