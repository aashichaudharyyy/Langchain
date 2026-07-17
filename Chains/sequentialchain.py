from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary report from the following text. \n {text}",
    input_variables=['text']
)

chain = prompt1 | model | parser | prompt2 | model | parser

res = chain.invoke({'topic':'Unemployement in India'})

print(res)
chain.get_graph().print_ascii()