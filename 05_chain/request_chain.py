from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
# from langchain.chains.llm import LLMChain
# from langchain_community.chains.llm_requests import LLMRequestsChain

# from langchain.schema.output_parser import StrOutputParser
# from langchain.callbacks.tracers import ConsoleCallbackHandler

from langchain_community.utilities import TextRequestsWrapper
# from langchain_core.runnables import RunnableParallel


chat = ChatOpenAI()
requests_wrapper = TextRequestsWrapper()

# GETリクエストを実行する関数を定義
def fetch_data(url):
    response = requests_wrapper.get(url)
    return {"requests_result": response}

# # 中間ステップで結果をマージする関数を定義
# def merge_inputs(inputs):
#     return {
#         "inputs_1": inputs[1]["query"],
#         "inputs_2": inputs[0]["requests_result"]
#     }

prompt = PromptTemplate(
    input_variables=[
        "query",
        "requests_result"],
    template="""以下の文章を元に質問に答えてください。
文章: {requests_result}
質問: {query}""",
)

# LLMへの入力用のプロンプトテンプレート
# input_variablesに設定したキーを前のRunnableから受け取れる。キー名は任意（今回は"response_content"としている。）
llm_prompt = PromptTemplate(
    input_variables=["response_content"],
    template="以下のレスポンスを要約してください。\n\n{response_content}"
)

# チェーンを定義
chain = RunnableSequence(
    # lambda input: input, 
    # fetch_data,
    # merge_inputs,
    prompt,  # PromptTemplateとしてRunnableを使用
    llm_prompt,  # PromptTemplateとしてRunnableを使用
    chat  # ChatOpenAIのインスタンスとしてcallableを使用
)

# 実行例
url = "https://www.jma.go.jp/bosai/forecast/data/overview_forecast/130000.json"
inputs = {
    "query": "東京の天気について教えて",
    "requests_result": fetch_data(url),
}

response = chain.invoke(inputs)
print(response)