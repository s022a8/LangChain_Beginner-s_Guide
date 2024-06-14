from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

output_parser = CommaSeparatedListOutputParser()

chat = ChatOpenAI(
    model="gpt-3.5-turbo",
)

result = chat.invoke(
    [
        HumanMessage(content="Appleが開発した代表的な製品を3つ教えてください"),
        HumanMessage(content=output_parser.get_format_instructions())
    ]
)

output = output_parser.parse(result.content)

for item in output:
    print("代表的な製品 => " + item)