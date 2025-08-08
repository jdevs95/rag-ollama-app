import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama

# Conversation history storage (in-memory)
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = {}

# Model names for Ollama
EMBEDDING_MODEL_NAME = "nomic-embed-text"
INFERENCE_MODEL_NAME = "llama3"
VECTORDB_DIR = "ollama_rag_db"

st.title("Local RAG App with Ollama (Llama3)")
st.write("Upload a PDF, ask questions, and get answers using local Llama3!")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")




if uploaded_file:
    with st.spinner("Uploading PDF..."):
        pdf_path = os.path.join("temp_" + uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

    st.success(f"Uploaded {uploaded_file.name}")

    with st.spinner("Loading and chunking PDF..."):
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
    # Increased chunk size and overlap for better context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=1500)
    splits = text_splitter.split_documents(pages)

    with st.spinner("Embedding chunks and storing in vector DB..."):
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
        vectorDB = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=VECTORDB_DIR)

    # Summarization section (automatic)
    st.header("Document Summary")
    full_text = "\n".join([page.page_content for page in pages])
    sq3r_prompt = (
            "You're a study assistant. Carefully read the following document and apply the SQ3R method (Survey, Question, Read, Recite, Review) behind the scenes.\n\n"
            "Your goal is to write a natural, flowing summary that:\n\n"
            "Gives an overview of what the document is about\n"
            "Introduces key questions the text answers (implicitly or explicitly)\n"
            "Covers the most important insights and ideas from the text\n"
            "Helps the reader understand the relevance and flow of the content before they study it in depth\n\n"
            "⚠️ Do not structure your output using the words 'Survey', 'Question', etc. Just produce a smooth, well-written summary based on that method.\n\n"
            "Document:\n" + full_text + "\nSummary:"
        )
    with st.spinner("Summarizing document with Llama3..."):
            response = ollama.chat(model=INFERENCE_MODEL_NAME, messages=[{'role':'user','content': sq3r_prompt}])
            summary = response['message']['content']
    st.write(summary)

    # Q&A section (independent)
    st.header("Ask a Question")
    question = st.text_input("Your question:")
    doc_key = uploaded_file.name
    if question:
        with st.spinner("Retrieving relevant context and getting answer from Llama3..."):
            retriever = vectorDB.as_retriever(search_type='mmr')
            retrieved_docs = retriever.invoke(question)
            context = "\n".join(doc.page_content for doc in retrieved_docs)
            # Less restrictive prompt
            qa_prompt = (
                "You are a helpful assistant. Answer the user's question using the information provided in the following document(s). "
                "If the answer is not present, do your best to infer it from the context.\n\n"
                f"Document(s):\n{context}\n\nQuestion: {question}\nAnswer:"
            )
            response = ollama.chat(model=INFERENCE_MODEL_NAME, messages=[{'role':'user','content': qa_prompt}])
            answer = response['message']['content']
        st.subheader("Answer")
        st.write(answer)
        # Store Q&A in conversation history for this document
        history = st.session_state['conversation_history'].setdefault(doc_key, [])
        history.append((question, answer))

    # Show conversation history for this document
    st.header("Conversation History")
    history = st.session_state['conversation_history'].get(doc_key, [])
    if history:
        for idx, (q, a) in enumerate(history[::-1], 1):
            st.markdown(f"**Q{len(history)-idx+1}:** {q}")
            st.markdown(f"**A:** {a}")
            st.write("---")
else:
    st.info("Please upload a PDF to begin.")
