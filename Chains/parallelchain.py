from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)

llm2 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Generate short and simple notes on the following text. \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text. \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document. \n {notes} and {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """ Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is an AI architecture that combines the reasoning abilities of Large Language Models (LLMs) with external knowledge sources. Traditional LLMs generate responses based only on the information they learned during training, which means they may produce outdated or incorrect answers when asked about recent events or domain-specific knowledge. RAG addresses this limitation by retrieving relevant information from external documents before generating a response.

A typical RAG pipeline consists of several stages. First, documents such as PDFs, websites, or databases are collected and converted into smaller chunks. These chunks are transformed into numerical vector representations called embeddings using an embedding model. The embeddings are then stored in a vector database such as FAISS or Chroma.

When a user submits a query, the query is also converted into an embedding. A similarity search is performed in the vector database to retrieve the most relevant document chunks. These retrieved chunks are combined with the user's question and provided to the LLM as additional context. As a result, the model generates responses that are more accurate, relevant, and grounded in the retrieved information rather than relying solely on its internal knowledge.

RAG offers several advantages. It reduces hallucinations, provides access to up-to-date information without retraining the model, and enables organizations to build AI applications using private or proprietary data. However, the quality of a RAG system depends heavily on document chunking strategies, embedding quality, retrieval accuracy, and prompt design.

Common applications of RAG include enterprise chatbots, document question-answering systems, legal assistants, medical knowledge systems, customer support automation, research assistants, and educational tutoring platforms. As the field of Generative AI evolves, RAG has become one of the most widely adopted architectures for building reliable, scalable, and production-ready AI applications.
"""

res = chain.invoke({'text':text})

print(res)

chain.get_graph().print_ascii()