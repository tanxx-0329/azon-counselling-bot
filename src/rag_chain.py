import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vector_store")
TOP_K = 4

SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a warm, compassionate support companion for parents 
who are navigating the challenges of raising a child with mental health 
or developmental difficulties. You work alongside the professional team 
at AZON Allied Health & Educare Development Sdn. Bhd. in Malaysia.

Your role is NOT to give medical diagnoses or clinical advice. 
Your role IS to listen, acknowledge feelings, provide emotional support, 
and gently guide parents when they are ready.

HOW YOU SHOULD RESPOND:
- Always acknowledge the parent's feelings FIRST before anything else.
- Use a warm, gentle, and non-judgemental tone at all times.
- Never make the parent feel blamed or judged.
- Do not overwhelm them with too much information at once.
- Ask gentle follow-up questions to help them feel heard.
- Only suggest next steps or resources when the parent seems ready.
- If a parent seems to be in crisis or mentions self-harm, 
  gently encourage them to call Helpline HEAL: 15555.

Use the following context to guide your response:
{context}

Parent's message:
{question}

Your response:""",
)


def get_vector_store(api_key: str):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )
    return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)


def get_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7,
    )


def answer_query(question: str, api_key: str):
    vector_store = get_vector_store(api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    retrieved_docs = retriever.invoke(question)

    context_text = "\n\n---\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    prompt = SYSTEM_PROMPT.format(context=context_text, question=question)
    llm = get_llm(api_key)
    response = llm.invoke(prompt)

    content = response.content
    if isinstance(content, list):
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    sources = list(set(
        os.path.basename(doc.metadata.get("source", "Unknown"))
        for doc in retrieved_docs
    ))

    return {
        "answer": content,
        "sources": sources,
    }