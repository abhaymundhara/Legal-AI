import logging
import json
import http.client
import urllib.parse
import requests
import re
import time
import os
from dotenv import load_dotenv

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq  # Groq client library
from bs4 import BeautifulSoup  # Import BeautifulSoup

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CaseQueryAssistant')

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
    # Instantiate Groq client
    client = Groq(api_key=GROQ_API_KEY)

    # Define the messages payload
    messages = [
        {
            "role": "system",
            "content": (
                "You are a legal AI assistant specializing in analyzing legal case summaries. "
                "Your task is to provide detailed and comprehensive insights based solely on the information provided. "
                "Focus on extracting key points, interpreting relevant legal principles, and thoroughly addressing the user's query. "
                "Include necessary context and avoid speculative statements; prioritize clarity and depth."
            ),
        },
        {
            "role": "user",
            "content": f"Query: {question}\n\nRelated case summaries:\n\n{related_case_summaries}",
        },
    ]

    try:
        # Create the completion request
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=0.6,
            max_tokens=1500,
            top_p=0.95,
            stream=True,
            stop=None,
        )

        # Collect and build the response from chunks
        answer = ""
        for chunk in completion:
            delta = chunk.choices[0].delta.content or ""
            answer += delta

        return answer.strip()

    except Exception as e:
        logger.error(f"query_ai_model: Error while querying AI: {str(e)}")
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
        
        # Initialize the Hugging Face API URL
        self.huggingface_api_url = HUGGINGFACE_API_URL
        
        # Hugging Face headers
        self.hf_headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
        }

    def clean_text(self, text):
        """Cleans the extracted text by removing HTML tags and excessive whitespace."""
        soup = BeautifulSoup(text, "html.parser")
        cleaned_text = soup.get_text(separator=" ", strip=True)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        return cleaned_text.strip()
    
    def summarize(self, text, max_length=200, min_length=100):
        """
        Uses Hugging Face Inference API to summarize text.
        Retries if the model is loading.
        """
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            self.logger.warning("summarize: Cleaned text is empty. Skipping summarization.")
            return ""
        
        truncated_text = cleaned_text[:1024]  # Truncate to 1024 characters
        payload = {
            "inputs": truncated_text,
            "parameters": {
                "max_length": max_length,
                "min_length": min_length,
                "truncation": True,
            },
        }

        max_retries = 5  # Maximum number of retries
        retry_delay = 60  # Wait time in seconds between retries
        attempt = 0

        while attempt < max_retries:
            try:
                response = requests.post(self.huggingface_api_url, headers=self.hf_headers, json=payload)

                if response.status_code == 503:  # Model is loading
                    estimated_time = response.json().get("estimated_time", retry_delay)
                    self.logger.info(f"summarize: Model is loading. Retrying in {estimated_time:.1f} seconds...")
                    time.sleep(min(estimated_time, retry_delay))
                    attempt += 1
                    continue

                if response.status_code != 200:
                    self.logger.error(f"summarize: Hugging Face API error {response.status_code}: {response.text}")
                    return f"Error: Unable to summarize due to API error: {response.status_code}"

                summary = response.json()
                if isinstance(summary, list) and "summary_text" in summary[0]:
                    return summary[0]["summary_text"]

                self.logger.error("summarize: Unexpected summary format received.")
                return "Error: No summary generated."

            except Exception as e:
                self.logger.error(f"summarize: Exception during summarization: {e}")
                attempt += 1
                time.sleep(retry_delay)

        # Return error if all retries fail
        self.logger.error("summarize: Max retries reached. Failed to summarize text.")
        return "Error: Summarization failed after multiple attempts."


    def fetch_case(self, case_input):
        """Fetches case details based on user input."""
        self.logger.info(f"fetch_case: Fetching cases for '{case_input}'")
        doc_ids = self.fetch_all_docs(case_input)
        
        if not doc_ids:
            self.logger.warning("fetch_case: No documents found.")
            return []

        cases = []
        for docid in doc_ids:
            doc = self.fetch_doc(docid)
            if not doc:
                continue

            title = doc.get("title", "No Title")
            case_text = doc.get("doc", "")
            cleaned_text = self.clean_text(case_text)

            if cleaned_text:
                cases.append({"docid": docid, "title": title, "cleaned_text": cleaned_text})
        
        return cases

    def fetch_all_docs(self, query):
        """Fetches all document IDs related to a query."""
        self.logger.info(f"fetch_all_docs: Searching documents for query '{query}'")
        doc_ids = []
        for pagenum in range(self.maxpages):
            encoded_query = urllib.parse.quote_plus(query)
            url = f'/search/?formInput={encoded_query}&pagenum={pagenum}'
            response = self.call_api(url)
            
            if not response:
                break

            try:
                data = json.loads(response)
                for doc in data.get("docs", []):
                    doc_ids.append(doc.get("tid"))
            except json.JSONDecodeError:
                break

        return doc_ids

    def fetch_doc(self, docid):
        """Fetches a document by its ID."""
        self.logger.info(f"fetch_doc: Fetching document {docid}")
        url = f'/doc/{docid}/'
        response = self.call_api(url)

        if not response:
            return None

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None

    def call_api(self, url):
        """Helper function to make API requests."""
        connection = http.client.HTTPSConnection(self.basehost)
        connection.request("POST", url, headers=self.headers)
        response = connection.getresponse()
        if response.status != 200:
            return None
        return response.read()

def get_text_chunks(text):
    """Splits text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    """Creates a vector store using FAISS."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, "faiss_index")
    with open("texts.json", "w") as f:
        json.dump(text_chunks, f)

def user_input(user_question):
    """Handles user question and generates answers."""
    faiss_index = faiss.read_index("faiss_index")
    with open("texts.json", "r") as f:
        text_chunks = json.load(f)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_embedding = model.encode([user_question])
    distances, indices = faiss_index.search(question_embedding, k=3)
    related_chunks = [text_chunks[i] for i in indices[0]]
    combined_context = " ".join(related_chunks)
    st.write(query_ai_model(user_question, combined_context))

def main():
    st.title("LegalLexa - AI Legal Assistant")
    st.write("Search and analyze legal cases.")
    ikapi = IKApi(maxpages=5)
    case_input = st.text_input("Enter Case Name or Query")
    if st.button("Search"):
        cases = ikapi.fetch_case(case_input)
        if cases:
            st.write(f"Found {len(cases)} cases.")
            selected_case = cases[0]
            text_chunks = get_text_chunks(selected_case["cleaned_text"])
            get_vector_store(text_chunks)
            question = st.text_input("Ask a question about the case")
            if question:
                user_input(question)

if __name__ == "__main__":
    main()
