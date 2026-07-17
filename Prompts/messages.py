from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
    "max_new_tokens": 700,
    "do_sample": False,
    "return_full_text": False,
    "repetition_penalty": 1.1,}
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me about LangChain")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)