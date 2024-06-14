from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI(
    model="gpt-3.5-turbo",
)

prompt = PromptTemplate(
    template="{product}はどこの会社が開発した製品ですか？",
    input_variables=[
        "product"
    ]
)
prompt_json = prompt.save("./02_model_io/prompt.json")