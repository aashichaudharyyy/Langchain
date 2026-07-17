
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
    "temperature": 0.2,
    "max_new_tokens": 700,
    "do_sample": False,
    "return_full_text": False,
    "repetition_penalty": 1.1,
}
)

model = ChatHuggingFace(llm=llm)

st.header("Research Paper Summarization Tool using Langchain")

paper_input = st.selectbox("Select Research Paper", ["Attention is All You Need", "BERT: Pre-training of Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])
style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly","Technical", "Code-Oriented","Mathematical"])
length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (6-10 paragraphs)"])

template = load_prompt("template.json")

# prompt = template.invoke({
#    'paper_input':paper_input,
#    'style_input':style_input,
#    'length_input':length_input
# })

if st.button("summarize"):
    #result = model.invoke(prompt)
    chain = template | model
    result = chain.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
    })
    st.write(result.content)

# Chain allows you to use invoke function only once rather than again and again - thw whole logic behind langchain.
