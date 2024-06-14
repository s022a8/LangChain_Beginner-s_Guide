from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate

examples = [
    {
        "input": "LangChainはChatGPT・Large Language Model(LLM)の実利用をより柔軟に簡易に行うためのツール群です",
        "output": "LangChainは、ChatGPT・Large Language Model(LLM)の実利用をより柔軟に、簡易に行うためのツール群です。"
    }
]

prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="入力: {input}\n出力: {output}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=prompt,
    
    prefix="以下の句読点の抜けた入力に句読点を追加してください。追加して良い句読点は「、」「。」のみです。他の句読点は追加しないでください。",
    suffix="入力: {input_string}\n出力:",
    input_variables=["input_string"],
)

llm = OpenAI(model="gpt-3.5-turbo-instruct")
formatted_prompt = few_shot_prompt.format(
    input_string="私はさまざまな機能がモジュールとして提供されているLangChainを使ってアプリケーションを開発しています"
)

result = llm.invoke(formatted_prompt)
print("formatted_prompt: ", formatted_prompt)
print("result: ", result)