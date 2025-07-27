import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import asyncio # Import asyncio
from google.api_core import exceptions # Import exceptions from google.api_core

# Set GRPC_POLL_STRATEGY to "poll" to potentially resolve asyncio event loop issues
os.environ["GRPC_POLL_STRATEGY"] = "poll"

# Your Google API Key
GOOGLE_API_KEY = "AIzaSyBfxXXypKxT0-SOzncW5m153D75r-kLRLA"

def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Splits the given text into smaller, manageable chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Creates a vector store (FAISS) from text chunks using Google Generative AI Embeddings.
    """
    # Ensure an asyncio event loop is available for the current thread
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index") # Save the index locally for persistence
    except exceptions.NotFound as e:
        st.error(f"Error: Could not access the embedding model. Please check your Google API Key and ensure the 'embedding-001' model is available and accessible. Details: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during vector store creation: {e}")
        st.stop()


def get_conversational_chain():
    """
    Defines the conversational chain for question answering using Gemini Pro.
    """
    # Ensure an asyncio event loop is available for the current thread
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in the provided context, just say, "Answer is not available in the context." Do not try to make up an answer.
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    """
    
    try:
        # Changed model from "gemini-pro" to "gemini-1.5-flash"
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except exceptions.NotFound as e:
        st.error(f"Error: Could not access the Gemini 1.5 Flash model. Please check your Google API Key and ensure the 'gemini-1.5-flash' model is available and accessible. Details: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during conversational chain setup: {e}")
        st.stop()


def user_input(user_question):
    """
    Processes the user's question, retrieves relevant documents, and generates an answer.
    """
    # Ensure an asyncio event loop is available for the current thread
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    except exceptions.NotFound as e:
        st.error(f"Error: Could not initialize embeddings for user input. Please check your Google API Key. Details: {e}")
        return
    except Exception as e:
        st.error(f"An unexpected error occurred during embeddings initialization: {e}")
        return
    
    # Check if the FAISS index exists
    if not os.path.exists("faiss_index"):
        st.warning("Please upload and process PDF documents first to create the FAISS index.")
        return

    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except Exception as e:
        st.error(f"Error loading FAISS index or performing similarity search: {e}")
        return

    chain = get_conversational_chain() # This function now handles its own errors

    if chain: # Only proceed if chain was successfully created
        try:
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            st.write("Reply: ", response["output_text"])
        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")


def main():
    st.set_page_config(page_title="PDF RAG App", layout="wide")
    st.header("Chat with Multiple PDFs using Gemini Pro 📚") # Title remains "Gemini Pro" but uses 1.5 Flash

    with st.sidebar:
        st.title("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF files here and click on 'Process Documents'",
            accept_multiple_files=True,
            type=["pdf"]
        )
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Documents Processed and Vector Store Created!")
            else:
                st.warning("Please upload at least one PDF document.")

    user_question = st.text_input("Ask a Question from the PDF Documents:")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
