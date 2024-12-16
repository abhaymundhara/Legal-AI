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

# INDIANKANOON_API_TOKEN = st.secrets["indiankanoon"]["INDIANKANOON_API_TOKEN"]
INDIANKANOON_API_TOKEN = os.getenv("INDIANKANOON_API_TOKEN")
# HUGGINGFACE_API_TOKEN = st.secrets["huggingface"]["HUGGINGFACE_API_TOKEN"]
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
# API_URL = st.secrets["openai"]["API_URL"]
# OPENAI_API_KEY = st.secrets["openai"]["OPENAI_API_KEY"]
# OPENAI_ENDPOINT = st.secrets["openai"]["OPENAI_ENDPOINT"]
#GROQ_API_KEY =  st.secrets["GROQ"]["GROQ_API"]
# GROQ_API_KEY =  "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"
GROQ_API_KEY =  os.getenv("GROQ_API_KEY")
# OPENAI_HEADERS = {
#     "Content-Type": "application/json",
#     "api-key": OPENAI_API_KEY,
# }


'''def query_ai_model(question, related_case_summaries):
    payload = {
        "messages": [
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
                    + related_case_summaries[:100]
                ),
            },
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 1000,
    }

    try:
        response = requests.post(OPENAI_ENDPOINT, headers=OPENAI_HEADERS, json=payload)
        response.raise_for_status()
        answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No answer found.")
    except requests.RequestException as e:
        #return f"Error while querying the AI: {str(e).split(':')[0:2]}{payload}"
        return f"Error while querying the AI: {str(e)}"
    return answer'''


from groq import Groq

def query_ai_model(question, related_case_summaries):
    # Instantiate Groq client
    client = Groq(api_key = GROQ_API_KEY)

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
        self.huggingface_api_url = API_URL
        self.hf_headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
        }

    def clean_text(self, text):
        text = re.sub(r"<[^>]+>", " ", text)  #  HTML
        text = re.sub(r"\s+", " ", text)  #  multiple spaces
        return text.strip()

    def truncate_text(self, text, max_tokens=1024):
        words = text.split()
        return " ".join(words[:max_tokens])

    def split_text_into_chunks(self, text, max_tokens=1024):
        words = text.split()
        for i in range(0, len(words), max_tokens):
            yield " ".join(words[i:i + max_tokens])

    def summarize(self, text, max_length=150, min_length=50):
        """
        Uses Hugging Face Inference API to summarize text.
        Ensures text length stays within model limits.
        """
        cleaned_text = self.clean_text(text)
        truncated_text = cleaned_text[:1024]  

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
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ikapi = IKApi(maxpages=5)

    query = "road accident cases"
    doc_ids = ikapi.fetch_all_docs(query)

    if not doc_ids:
        print(f"No documents found for query: {query}")
    else:
        print(f"Found {len(doc_ids)} documents for query: {query}")
    
    all_summaries = []

    # Iterate through all document IDs
    for docid in doc_ids[:2]:
        case_details = ikapi.fetch_doc(docid)

        if not case_details:
            print(f"Failed to fetch details for document ID: {docid}")
            continue

        title = case_details.get("title", "No Title")
        main_text = case_details.get("doc", "")
        cleaned_text = ikapi.clean_text(main_text)

        # Process chunks if text exceeds the limit
        chunks = list(ikapi.split_text_into_chunks(cleaned_text))
        summaries = []
        for chunk in chunks:
            summary = ikapi.summarize(chunk)
            if summary:
                summaries.append(summary)

        final_summary = " ".join(summaries)
        all_summaries.append(f"Title: {title}\nSummary: {final_summary}")

    # Combine all summaries into a single text
    combined_summary = "\n\n".join(all_summaries)
    
    print("Combined Summary of All Documents:")
    print(combined_summary)

   
    print(query_ai_model(query, combined_summary))
