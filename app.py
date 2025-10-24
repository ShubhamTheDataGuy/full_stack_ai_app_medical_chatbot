from flask import Flask, render_template, jsonify, request

from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *

import os

# ---------------------------
# Flask app initialization
# ---------------------------
app = Flask(__name__)
load_dotenv()

# ---------------------------
# Load API keys
# ---------------------------
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GEMINI_API_KEY = os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# ---------------------------
# Load embeddings and retriever
# ---------------------------
embeddings = download_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ---------------------------
# Initialize LLM model
# ---------------------------
chatmodel = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_tokens=3000,
    temperature=0.1,
    timeout=None,
    max_retries=2,
    max_completion_tokens=100,
)

# ---------------------------
# Create conversation memory
# ---------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ---------------------------
# Define custom prompt
# ---------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# ---------------------------
# Build Conversational Retrieval Chain
# ---------------------------
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=chatmodel,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# ---------------------------
# Flask routes
# ---------------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)

    # Invoke RAG chain with memory
    response = rag_chain.invoke({"question": msg})

    print("Response:", response["answer"])
    return str(response["answer"])


# ---------------------------
# Run app
# ---------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
