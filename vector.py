from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ConfluenceLoader
import pytesseract

import reportlab.graphics.renderPM as renderPM
renderPM._renderPMBackend = "bitmap"

# Confluence API token, TODO: Make modifiable
CONFLUENCE_API_TOKEN = "ATATT3xFfGF0biwfWNN6mqTtJ2Lp1ZipEjeH5IDpE6WN8Xes5rJgFzMggQka09sy8PAp6t_9P6QXHQtrf4eili8mjqENmD2Do4oGvBAG1UrZb5_9R-Qv-UXtxoQE8e9JCcowMN2YgCv9EuayYCiOqhFZJGyI_5m5utQJEm1hJFm7v5c06JnN2zE=BFAE4785"

def createRetriever():
    loader = ConfluenceLoader(
        url="https://europeana.atlassian.net/wiki/",
        username="jordenjessevs@gmail.com", 
        keep_markdown_format= True,
        api_key=CONFLUENCE_API_TOKEN, cloud= True, space_key="EF", include_attachments=True)

    # Necessary for ConfluenceLoader? 
    pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

    # Load the documents
    docs = loader.load()

    # Initialize the chunker, can choose different chunkers here. TODO: Make modifiable
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])

    # Chunk the cleaned documents. 
    documentswiki = text_splitter.split_documents(docs)

    # Use free ollama embedding, should be changed for testing purposes TODO: Make modifiable
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    
    # Add documents to db.
    vectorstore = Chroma.from_documents(
        documents=documentswiki,
        embedding=embedding,
        persist_directory="chroma_db" 
    )

    # chromadb retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":6})

    return retriever
