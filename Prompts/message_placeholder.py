from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#chat template
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# load chat history
chat_history = []
with open('Prompts/chathistory.txt') as f:
    chat_history.extend(f.readlines())

# create prompt

query = input("Query:")
prompt = chat_template.invoke({'chat_history':chat_history,'query':query})

print(prompt)