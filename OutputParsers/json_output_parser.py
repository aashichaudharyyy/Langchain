from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

parser = JsonOutputParser()

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template='Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt = template.format()
res = model.invoke(prompt)

final_res = parser.parse(res.content)

print(res)
print(final_res)

# now with chains

chain = template | model | parser

result = chain.invoke({})
print(result)