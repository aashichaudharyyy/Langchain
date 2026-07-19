from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence, RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

def word_count(text):
    return len(text.split())

chain = RunnableSequence(prompt1, model, parser, RunnableParallel({
    'joke': RunnablePassthrough(),
    'words': RunnableLambda(word_count)
}))

result = chain.invoke({"topic":"Dog"})

print(result)