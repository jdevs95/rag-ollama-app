
import openai
import os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_qa_answer(context, question, model_name="gpt-4o"):
    prompt = (
        "You are a helpful assistant. ONLY answer questions using the information provided in the following document(s). "
        "If the answer is not present or cannot be inferred from the document(s), reply: 'Sorry, the answer is not available in the provided document(s).'\n\n"
        f"Document(s):\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512
    )
    answer = response.choices[0].message.content.strip()
    return answer, "OpenAI gpt-4o"

def summarize_text(text, model_name="gpt-4o", chunk_size=2000, max_length=4096):
    prompt_template = (
        "You're a study assistant. Carefully read the following document and apply the SQ3R method (Survey, Question, Read, Recite, Review) behind the scenes.\n\n"
        "Your goal is to write a natural, flowing summary that:\n\n"
        "Gives an overview of what the document is about\n"
        "Introduces key questions the text answers (implicitly or explicitly)\n"
        "Covers the most important insights and ideas from the text\n"
        "Helps the reader understand the relevance and flow of the content before they study it in depth\n\n"
        "⚠️ Do not structure your output using the words 'Survey', 'Question', etc. Just produce a smooth, well-written summary based on that method.\n\n"
        "Document:\n{text}\nSummary:"
    )
    summaries = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        prompt = prompt_template.format(text=chunk)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_length
        )
        summary = response.choices[0].message.content.strip()
        summaries.append(summary)
    return " ".join(summaries), "OpenAI gpt-4o"