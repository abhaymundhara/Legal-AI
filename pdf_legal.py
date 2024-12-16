import logging
import json
import http.client
import urllib.parse
import requests
import re
import time
import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from groq import Groq  # Groq client library

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('LegalLexa')

# Load API tokens from environment variables
INDIANKANOON_API_TOKEN = os.getenv("INDIANKANOON_API_TOKEN")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define Hugging Face API URL for summarization
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

def truncate_text_to_fit(text, max_chars):
    """Truncates text to fit within a maximum character limit."""
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text

def query_ai_model(question, related_case_summaries):
    """
    Queries the Groq LLaMA model to generate an answer based on the question and context.
    """
    # Instantiate Groq client inside the function as per your original code
    client = Groq(api_key=GROQ_API_KEY)

    # Define the messages payload
    messages = [
        {
            "role": "system",
            "content": (
                "You are a legal AI assistant specializing in analyzing legal case summaries. "
                "Your task is to provide concise, actionable insights based solely on the information provided. "
                "Focus on extracting key points, interpreting relevant legal principles, and addressing the user's query directly. "
                "Avoid boilerplate language, unnecessary context, or speculation; prioritize clarity and precision."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Query: {question}\n\n"
                "Related case summaries:\n\n"
                + related_case_summaries  # Ensure the summaries are not too long
            ),
        },
    ]

    try:
        # Create the completion request
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=0.95,
            stream=True,  # Enables streaming for incremental responses
            stop=None,
        )

        # Collect and build the response from chunks
        answer = ""
        for chunk in completion:
            delta = chunk.choices[0].delta.content or ""
            answer += delta

        # Return the final response
        return answer.strip()

    except Exception as e:
        # Handle exceptions and return error message
        logger.error(f"Error while querying AI: {str(e)}")
        return f"Error while querying AI: {str(e)}"

class IKApi:
    def __init__(self, maxpages=1):
        self.logger = logging.getLogger('ikapi')
        self.headers = {
            'Authorization': f'Token {INDIANKANOON_API_TOKEN}',
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0'
        }
        self.basehost = 'api.indiankanoon.org'
        self.maxpages = min(maxpages, 100)
        self.huggingface_api_url = HUGGINGFACE_API_URL
        self.hf_headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
        }

    def clean_text(self, text):
        text = re.sub(r"<[^>]+>", " ", text)  # Remove HTML tags
        text = re.sub(r"\s+", " ", text)     # Replace multiple spaces with single space
        return text.strip()

    def truncate_text(self, text, max_tokens=1024):
        words = text.split()
        return " ".join(words[:max_tokens])

    def split_text_into_chunks(self, text, max_tokens=1024):
        words = text.split()
        for i in range(0, len(words), max_tokens):
            yield " ".join(words[i:i + max_tokens])

    def summarize(self, text, max_length=100, min_length=50):
        """
        Uses Hugging Face Inference API to summarize text.
        Ensures text length stays within model limits.
        """
        cleaned_text = self.clean_text(text)
        truncated_text = cleaned_text[:1024]  # Truncate to 1024 characters

        payload = {
            "inputs": truncated_text,
            "parameters": {
                "max_length": max_length,
                "min_length": min_length,
                "truncation": True
            }
        }

        try:
            response = requests.post(self.huggingface_api_url, headers=self.hf_headers, json=payload)

            # Retry if model is loading
            if response.status_code == 503:
                estimated_time = response.json().get("estimated_time", 10)
                self.logger.info(f"Model is loading. Retrying in {estimated_time} seconds...")
                time.sleep(estimated_time)
                return self.summarize(text, max_length, min_length)

            if response.status_code != 200:
                self.logger.error(f"Hugging Face API error {response.status_code}: {response.text}")
                return f"Error: Unable to summarize due to API error: {response.status_code}"

            summary = response.json()
            if isinstance(summary, list) and "summary_text" in summary[0]:
                return summary[0]["summary_text"]
            return summary.get("summary_text", "Error: No summary generated.")

        except Exception as e:
            self.logger.error(f"Exception during summarization: {e}")
            return f"Error: {str(e)}"

    def fetch_doc(self, docid):
        """Fetch document by ID."""
        url = f'/doc/{docid}/'
        connection = http.client.HTTPSConnection(self.basehost)
        connection.request("POST", url, headers=self.headers)
        response = connection.getresponse()
        if response.status != 200:
            self.logger.warning(f"Failed to fetch document {docid}. HTTP {response.status}: {response.reason}")
            return None
        return json.loads(response.read())

    def fetch_all_docs(self, query):
        """
        Fetches all document IDs related to a query from Indian Kanoon.
        Args:
            query (str): The search query.
        Returns:
            list: A list of document IDs (docid) matching the query.
        """
        doc_ids = []
        pagenum = 0

        while pagenum < self.maxpages:
            encoded_query = urllib.parse.quote_plus(query)
            url = f'/search/?formInput={encoded_query}&pagenum={pagenum}&maxpages=1'
            results = self.call_api(url)

            if not results:
                self.logger.warning(f"No results for query '{query}' on page {pagenum}")
                break

            try:
                obj = json.loads(results)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse results for query '{query}' on page {pagenum}")
                break

            if 'docs' not in obj or not obj['docs']:
                break

            for doc in obj['docs']:
                docid = doc.get('tid')
                if docid:
                    doc_ids.append(docid)

            pagenum += 1

        return doc_ids

    def call_api(self, url):
        connection = http.client.HTTPSConnection(self.basehost)
        connection.request('POST', url, headers=self.headers)
        response = connection.getresponse()
        results = response.read()
        return results

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

def get_text_chunks(text):
    """Splits text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Generates embeddings for text chunks using Sentence Transformers and
    creates a FAISS vector store for similarity search.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and efficient model
    embeddings = model.encode(text_chunks, convert_to_tensor=False)

    # Create FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    # Save FAISS index and texts
    faiss.write_index(faiss_index, "faiss_index")
    with open("texts.json", "w") as f:
        json.dump(text_chunks, f)

def get_conversational_chain():
    """
    Sets up the QA chain using Groq's LLaMA model.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
    If the answer is not in the provided context, just say, "Answer is not available in the context." 
    Don't provide the wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Initialize the QA chain with LangChain's load_qa_chain
    chain = load_qa_chain(
        llm=None,  # We will handle LLM interaction manually with Groq
        chain_type="stuff",
        prompt=prompt
    )

    return chain

def user_input(user_question):
    """
    Handles the user question input, performs similarity search, and generates an answer.
    """
    # Load FAISS index and texts
    faiss_index = faiss.read_index("faiss_index")
    with open("texts.json", "r") as f:
        text_chunks = json.load(f)

    # Initialize embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_embedding = model.encode([user_question], convert_to_tensor=False)

    # Perform similarity search
    D, I = faiss_index.search(question_embedding, k=3)  # Reduced from 5 to 3
    related_docs = [text_chunks[idx] for idx in I[0]]

    # Combine related documents
    combined_context = "\n\n".join(related_docs)

    # Truncate the combined context if it's too long
    max_context_length = 2000  # Adjust based on model's capacity
    combined_context = truncate_text_to_fit(combined_context, max_context_length)

    # Generate answer using Groq's LLaMA model
    answer = query_ai_model(user_question, combined_context)

    # Display Q&A in Streamlit
    placeholder = st.empty()
    placeholder.subheader("Your Questions and Answers:")
    placeholder.write(f"**Question:** {user_question}")
    placeholder.write(f"**Answer:** {answer}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="LegalLexa - Chat PDF", layout="wide")
    st.markdown(
        """
        <style>
        body {
            background-color: #f2f2f2; /* Light grey color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.header("LegalLexa 👨‍🚀 - Your AI Legal Assistant")
    
    # User question input
    user_question = st.text_input("Ask a Question from the PDF Files")
    
    if user_question:
        user_input(user_question)
    
    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            type=["pdf"],
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    # Initialize IKApi with desired number of pages
                    ikapi = IKApi(maxpages=5)
                    
                    # Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if not raw_text:
                        st.error("No text found in the uploaded PDFs.")
                        return
                    
                    # Split text into chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # Create and save vector store
                    get_vector_store(text_chunks)
                    
                    st.success("Processing Complete! You can now ask questions about your PDFs.")

if __name__ == "__main__":
    main()
