import os
import zipfile
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# print(PyPDFLoader)
# print(RecursiveCharacterTextSplitter)
# print(FAISS)
# print(HuggingFaceEmbeddings)


#Extract zip files from data folder
DATA_PATH='Data/'

def extract_zip_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.zip'):
            zip_path=os.path.join(folder_path, filename)
            with zipfile.ZipFile(zip_path, 'r')as zip_ref:
                zip_ref.extractall(folder_path)
                print(f'Extracted: {filename}')



extract_zip_files(DATA_PATH)




#Step-1: load all pdfs (even inside subfolders)
def load_pdf_files(data_path):
    all_documents=[]
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path=os.path.join(root, file)
                loader=PyPDFLoader(pdf_path)
                documents = loader.load()
                all_documents.extend(documents)
    print(f"📄 Loaded {len(all_documents)} PDF pages.")
    return all_documents

documents=load_pdf_files(DATA_PATH)

# def load_pdf_files(data):
#     loader=DirectoryLoader(data,
#                            glob='*.pdf',
#                            loader_cls=PyPDFLoader)
#     documents=loader.load()
#     return documents

# documents=load_pdf_files(data=DATA_PATH)
# print('Length of PDF Pages', len(documents))

#Step: 2
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
print('Length of Text Chunks', len(text_chunks))

#Step:3
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embedding_model

embedding_model=get_embedding_model()

# Step 4: Vector embeddings in FAISS
DB_FAISS_PATH = 'vectorstore/db_faiss'

import os
if os.path.exists(DB_FAISS_PATH):
    print("Loading existing vectorstore...")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    print("Creating and saving new vectorstore...")
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)

# # Extra: Print Arabic content for inspection
# loader = PyPDFLoader("Data/CBUAE_EN_3920_VER1.pdf")
# documents = loader.load()

# # Print out some document content to check Arabic
# for i, doc in enumerate(documents[:5]):
#     print(f"\n----- Page {i + 1} -----\n")
#     print(doc.page_content)

# # Optional: similarity search query example
# query = "What is the car loan regulation?"
# results = db.similarity_search(query, k=3)

# print("\n=== Similarity Search Results ===\n")
# for i, res in enumerate(results):
#     print(f"Result {i + 1}:")
#     print(res.page_content)
#     print("-" * 40)
