from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from pydantic import BaseModel, Field, field_validator

chat = ChatOpenAI()

class Smartphone(BaseModel):
    release_date: str = Field(description="スマートフォンの発売日")
    screen_inches: float = Field(description="スマートフォンの画面サイズ(インチ)")
    os_installed: str = Field(description="スマートフォンにインストールされているOS")
    model_name: str = Field(description="スマートフォンのモデル名")
    
    @field_validator("screen_inches")
    def validate_screen_inches(cls, field):
        if field <= 0:
            raise ValueError("Screen inches must be a positive number")
        return field

parser = OutputFixingParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=Smartphone),
    llm=chat
)

result = chat.invoke([
    HumanMessage(content="Androidでリリースしたスマートフォンを1個挙げて"),
    HumanMessage(content=parser.get_format_instructions())
])
    
parsed_result = parser.parse(result.content)
print(f"発売日: {parsed_result.release_date}")
print(f"画面サイズ: {parsed_result.screen_inches}")
print(f"OS: {parsed_result.os_installed}")
print(f"モデル名: {parsed_result.model_name}")
    

