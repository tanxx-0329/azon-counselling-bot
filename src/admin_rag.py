import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

ADMIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an intelligent clinical assistant for therapists and administrators 
at AZON Allied Health & Educare Development Sdn. Bhd. in Malaysia.

Your role is to help therapists:
- Retrieve and summarise patient records and treatment history
- Organise and categorise appointment schedules
- Track patient progress over time
- Flag potential risks or concerns based on clinical notes
- Generate structured summaries for clinical use

HOW YOU SHOULD RESPOND:
- Be concise, professional, and clinically appropriate
- Present data in a clear, structured format when summarising
- When asked to categorise or organise, use clear headings
- Flag any risk indicators mentioned in the notes (e.g. self-harm, crisis)
- Always base your response on the data provided — do not fabricate

Available data:
{context}

Therapist's request:
{question}

Your response:""",
)


def admin_answer(question: str, api_key: str, db_context: str) -> dict:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3,
    )
    prompt = ADMIN_PROMPT.format(context=db_context, question=question)
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return {"answer": content}