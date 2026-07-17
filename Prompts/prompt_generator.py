from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:

Explanation Style: {style_input}

Explanation Length: {length_input}

1. Mathematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.

2. Analogies:
   - Use relatable analogies to simplify complex ideas.

3. Summary Guidelines:
   - Clearly explain the main objective of the paper.
   - Highlight the methodology used by the authors.
   - Mention the key contributions and findings.
   - Keep the explanation aligned with the selected explanation style.
   - Follow the selected explanation length.
   - Do not repeat the prompt or include unnecessary introductory text.
   - Do not generate information that is not present in the paper.

If certain information is not available in the paper, respond with:
"Insufficient information available" instead of guessing.

Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
    input_variables=[
        "paper_input",
        "style_input",
        "length_input",
    ],
)

template.save("template.json")


# Why to use PromptTemplate instead of f strings? -Chain working, -Reusable, -dynamic prompting instead of static prompt, -Default validation




