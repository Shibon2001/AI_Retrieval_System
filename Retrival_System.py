# Install required packages (Run in terminal or Colab)
# pip install langchain langchain-google-genai faiss-cpu sentence-transformers langchain-community langchain-text-splitters

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# =========================
# SET YOUR GOOGLE API KEY
# =========================
# Replace with your own Gemini API key
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

# =========================
# KNOWLEDGE BASE TEXT
# =========================
text = """
Python Tutorial
Last Updated : 8 May, 2026

Python is one of the most popular programming languages.
It is simple to use, beginner-friendly, and widely used in:

- Data Science
- Artificial Intelligence
- Automation
- Web Development
- Machine Learning

Python is known for:
- Clean syntax
- Readability
- Huge library ecosystem
- Cross-platform support

Popular libraries include:
- NumPy
- Pandas
- TensorFlow
- PyTorch
- Scikit-learn
- Flask
- Django
"""

# =========================
# SPLIT TEXT INTO CHUNKS
# =========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

# Create chunks/documents
docs = splitter.create_documents([text])

# =========================
# CREATE EMBEDDINGS MODEL
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# CREATE VECTOR DATABASE
# =========================
vectorstore = FAISS.from_documents(docs, embeddings)

# Save vector database locally (optional)
vectorstore.save_local("faiss_index")

# =========================
# CREATE RETRIEVER
# =========================
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# =========================
# FORMAT RETRIEVED DOCS
# =========================
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# =========================
# PROMPT TEMPLATE
# =========================
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use only the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
)

# =========================
# LOAD GEMINI MODEL
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

# =========================
# RAG FUNCTION
# =========================
def rag_answer(query):
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(query)

    # Format retrieved content
    context = format_docs(retrieved_docs)

    # Create final prompt
    full_prompt = prompt.format(
        context=context,
        question=query
    )

    # Generate response
    response = llm.invoke(full_prompt)

    return response.content

# =========================
# TEST QUERY
# =========================
if __name__ == "__main__":
    query = "What is Python used for?"

    answer = rag_answer(query)

    print("Question:", query)
    print("\nAnswer:")
    print(answer)
