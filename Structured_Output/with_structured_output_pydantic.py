from typing import Literal, Optional

from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
# from typing import Optional, TypedDict, Annotated
# from langchain_protocol import Literal
from pydantic import BaseModel, Field

load_dotenv()

# model = ChatOpenAI()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

#schema - data format

# class Review(TypedDict):
#     summary: str
#     sentiment: str

class Review(BaseModel):
    key_themes : list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos","neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(description="Write all the pros in a list")
    cons: Optional[list[str]] = Field(description="Write all the cons in a list")
    name: Optional[str] = Field(description="Write name of the reviewer")


# class Review(TypedDict):
#     key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
#     summary: Annotated[str, "A brief summary of the review"]
#     sentiment: Annotated[Literal["pos","neg"], "Return sentiment of the review either negative, positive or neutral"]
#     pros: Annotated[Optional[list[str]], "Write all the pros in a list"]
#     cons: Annotated[Optional[list[str]], "Write all the cons in a list"]
#     name: Annotated[Optional[str],"Write name of the reviewer"]


#structured_model = model.with_structured_output(Review)


#result = structured_model.invoke("The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.")
result = model.invoke("The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.")

print(result)


# structured output does work only with ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI and not huggingface

#problem with typed dict is -> data validation not possible much only representation & also doesn't bound to given types.