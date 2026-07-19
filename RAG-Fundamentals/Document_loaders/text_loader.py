from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="What is this document talk about? {text}",
    input_variables=['text']
)

parser = StrOutputParser()

loader = TextLoader('RAG-Fundamentals/Document_loaders/cricket.txt', encoding= 'UTF-8')

docs = loader.load()

chain = prompt1 | model | parser

result = chain.invoke({'text': docs[0].page_content})
print(result)