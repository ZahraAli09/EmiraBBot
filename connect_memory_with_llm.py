import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq  # NEW import
from dotenv import load_dotenv
import os

load_dotenv() 

# Step 1: Setup LLM (Mistral-7B via Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def load_llm():
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192",  # You can also try "llama3-70b-8192" or "gemma-7b-it"
        temperature=0.5,
        max_tokens=512
    )
    return llm

# Step 2: Create Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
You are a helpful, professional, and knowledgeable banking assistant specializing in the UAE banking system. Your role is to provide clear, accurate, and up-to-date information about:

- UAE Central Bank regulations
- Islamic banking and conventional banking in UAE
- Accounts (current, savings, salary)
- Credit cards, loans, and financing options
- KYC (Know Your Customer) requirements
- Bank transfers (local, international, WPS)
- Digital banking and mobile apps
- Fees, charges, and interest rates
- Banking for residents vs. non-residents
- Bank working hours and public holidays
- Account opening for individuals and businesses

Always tailor your responses using the following guidelines:

1. **Context-Aware:** Respond based on the banking laws, financial institutions, and practices specific to the United Arab Emirates (UAE).
2. **Concise Yet Informative:** Give to-the-point responses but provide links to official resources if more detail is needed.
3. **Professional Tone:** Maintain a respectful, formal tone with a customer-centric approach.
4. **Do Not Provide Legal or Investment Advice.**
5. **If Unsure:** If the user's question requires legal interpretation or up-to-date verification (like current interest rates), suggest checking with the relevant UAE bank or Central Bank website.

Context:
{context}

Question:
{question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

# Step 3: Load FAISS Vectorstore
DB_FAISS_PATH = 'vectorstore/db_faiss'
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Ask a question
user_query = input("Write Query Here: ")
response = qa_chain.invoke({"query": user_query})

# Step 6: Print results
print("\nRESULT:\n", response["result"])
print("\nSOURCE DOCUMENTS:\n", response["source_documents"]) 