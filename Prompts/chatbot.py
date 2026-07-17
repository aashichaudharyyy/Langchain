from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
    "max_new_tokens": 700,
    "do_sample": False,
    "return_full_text": False,
    "repetition_penalty": 1.1,
}
)

model = ChatHuggingFace(llm=llm)

chatHistory = [
    SystemMessage(content="You are a helpful AI assistant")
]

while True:
    user_input = input('You: ')
    chatHistory.append(HumanMessage(content=user_input)) 
    if user_input == 'exit':
        break
    result = model.invoke(chatHistory)
    chatHistory.append(AIMessage(content=result.content))
    print('AI: ',result.content) 

print(chatHistory)


