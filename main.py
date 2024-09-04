import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import sys
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import pinecone

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

# Configure Gemini API
import google.generativeai as genai
genai.configure(api_key=google_api_key)

# Initialize Pinecone
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

# Load the HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MPNet-base-v2")

print(pc.list_indexes().names())
index = pc.Index(pinecone_index_name)
vectordb = PineconeStore.from_existing_index(pinecone_index_name, embeddings)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided pdf documents, make sure to provide all the details, if the answer is not in
    provided document just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    docs = vectordb.similarity_search(user_question)
    chain = get_conversational_chain()
    print(docs)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    return response

def main():
    user_question = input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/add_todo', methods=['POST', 'GET'])
def add_todo():
    query = request.get_json()
    print(query)
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        sys.exit()
    result = user_input(query)
    print('Answer: ' + result['output_text'] + '\n')
    return {'message': result['output_text']}

print(__name__)
if __name__ == "__main__":
    app.run(debug=True)
