from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

#1st promopt -> detailed summary
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

#2nd prompt -> summary
template2 = PromptTemplate(
    template="Write a 5-line summaruy on thr following text. /n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

#without parser 
#chain1 = template1 | model
#res1 = chain1.invoke({'topic':'blackhole'})
#chain2 = template2 | model
#res2 = chain2.invoke({'text':res1.content})

result = chain.invoke({'topic':'blackhole'})

print(result)
