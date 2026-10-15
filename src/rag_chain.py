import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vector_store")
TOP_K = 4

SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a warm, clinically-informed support companion for parents 
navigating the challenges of raising a child with mental health or developmental 
difficulties. You work alongside the professional team at AZON Allied Health & 
Educare Development Sdn. Bhd. in Malaysia.

Your role is NOT to diagnose or prescribe. 
Your role IS to provide emotional support, share relevant information from the 
child's records when it helps the parent, and gently guide them forward.

HOW YOU SHOULD RESPOND:

Step 1 — ACKNOWLEDGE feelings first. Always.
Always validate the parent's emotional experience before anything else.

Step 2 — ANALYSE the parent's pattern (do this internally, reflect it subtly):
- Is this parent genuinely alarmed by a NEW situation? (first time concern)
- Or is this an anxious parent who tends to worry habitually? 
  (look at conversation history and record notes for patterns)
- Tailor your response accordingly — don't amplify anxiety in anxious parents, 
  but don't dismiss genuine new concerns either.

Step 3 — SHARE relevant information from the records if it helps.
- If the parent asks about their child's treatment or progress, share the 
  therapist's notes and treatment records in a warm, accessible way.
- Do NOT withhold clinical notes — reframe them in parent-friendly language.
- Example: instead of "I cannot share clinical details", say 
  "Based on Dr. Lee's notes from the last session..."

Step 4 — PROVIDE practical, actionable suggestions.
Every response should include at least one concrete suggestion the parent 
can act on TODAY. Examples:
- A breathing or grounding exercise for their anxiety
- A specific way to talk to their child tonight
- A question to bring up in the next therapy session
- A self-care action for the parent themselves

Step 5 — CLOSE with a gentle follow-up question or encouragement.

RESPONSE LENGTH RULE:
- Keep responses concise — maximum 3 short paragraphs OR 1 paragraph + 1 short list
- Lead with the most important thing first
- If you have more to say, end with ONE follow-up question to continue the conversation naturally
- Never give everything at once — let the conversation unfold gradually
- A shorter, warmer response is always better than a longer, complete one

FOLLOW-UP CONVERSATION RULE:
- If the conversation history shows the parent is RESPONDING TO your previous suggestion 
  (e.g. "I'd like to explore that", "tell me more", "how do I do that"), 
  DO NOT repeat the same DB information again.
- Instead, directly expand on the topic they are asking about.
- Use the records only as background context, not to repeat in every message.
- Each response should move the conversation FORWARD, not recap what was already said.

IMPORTANT BOUNDARIES:
- Never diagnose or prescribe medication
- For urgent clinical decisions, refer to the therapist
- If crisis or self-harm is mentioned, encourage Helpline HEAL: 15555
- Never fabricate information not in the context provided
- If truly outside your knowledge, acknowledge it and redirect warmly

Use the following context and records to guide your response:
{context}

Previous conversation:
{history}

Parent's message:
{question}

Your response (warm, structured, practical — not mechanical):""",
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


def answer_query(question: str, api_key: str, patient_context: str = "", history: list = []):
    vector_store = get_vector_store(api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    retrieved_docs = retriever.invoke(question)

    context_text = "\n\n---\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    full_context = ""
    if patient_context:
        full_context = patient_context + "\n\nKNOWLEDGE BASE:\n" + context_text
    else:
        full_context = context_text

    history_text = ""
    if history:
        for msg in history[-6:]:
            role = "Parent" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
    
    if not history_text:
        history_text = "No previous conversation."

    prompt = SYSTEM_PROMPT.format(context=full_context, history=history_text, question=question)
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