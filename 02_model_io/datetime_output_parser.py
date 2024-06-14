from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import DatetimeOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

output_parser = DatetimeOutputParser()

chat = ChatOpenAI(
    model="gpt-3.5-turbo",
)

prompt = PromptTemplate.from_template("{product}の発売日を教えて")

result = chat.invoke(
    [
        HumanMessage(content=prompt.format(product="iPhone8")),
        HumanMessage(content=output_parser.get_format_instructions())
    ]
)

output = output_parser.parse(result.content)

print(output)