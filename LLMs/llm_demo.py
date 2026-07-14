#Since OpenAI API is paid, i currently am not using it but this code here is for future reference.
#LLM take string and inout and string as output.

from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct')

result = llm.invoke("What is the capital of India")

print(result)

