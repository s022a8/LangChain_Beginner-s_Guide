# from langchain.chains.sequential import SimpleSequentialChain
# from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo")

write_prompt = PromptTemplate(
    template="{input}についての記事を書いてください。",
    input_variables=["input"]
)
translate_prompt = PromptTemplate(
    template="以下の文章を英語に翻訳してください。\n{input}",
    input_variables=["input"]
)

write_chain: RunnableSequence = write_prompt | chat
translate_chain: RunnableSequence = translate_prompt | chat

chain = RunnableSequence(
    write_chain,
    translate_chain
)

result = chain.invoke({"input" : "エレキギターの選び方"})
print(result)