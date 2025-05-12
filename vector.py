from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import ConfluenceLoader
from langchain_core.documents import Document
from dotenv import load_dotenv
import ollama
import os

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API token (or other environment variables)
api_token = os.getenv("CONFLUENCE_API_TOKEN")

import pytesseract

def createRetriever():
    loader = ConfluenceLoader(
        url="https://europeana.atlassian.net/wiki/",
        username="jordenjessevs@gmail.com", 
        #max_pages = "???"
    #    keep_markdown_format= True,
        api_key=api_token, cloud= True, space_key="EF", include_attachments=True)
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
        persist_directory="chroma_db_txt" 
    )

    # chromadb retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":6})

    return retriever

def createDBs():
    ollama.api_base = 'http://127.0.0.1:11434'
    loader_markdown = ConfluenceLoader(
        url="https://europeana.atlassian.net/wiki/",
        username="jordenjessevs@gmail.com", 
        keep_markdown_format= True,
        api_key=api_token, cloud= True, space_key="EF", include_attachments=True)

    loader_no_markdown = ConfluenceLoader(
        url="https://europeana.atlassian.net/wiki/",
        username="jordenjessevs@gmail.com", 
        api_key=api_token, cloud= True, space_key="EF", include_attachments=True)
    # Necessary for ConfluenceLoader? 
    pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

    # Load the documents
    docs_markdown = loader_markdown.load()
    docs_no_markdown = loader_no_markdown.load()

    # Initialize the chunker, can choose different chunkers here. TODO: Make modifiable
    rec_text_splitter_500 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    rec_text_splitter_1000 = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    #TODO: MarkdownTextSplitter or MarkdownHeaderTextSplitter
    #mark_text_splitter = MarkdownTextSplitter()
    headers_to_split_on = [("#", "Header 1"),("##", "Header 2"),("###", "Header 3")]
    mark_text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

    docs = []
    docs_rec_500_mark = rec_text_splitter_500.split_documents(docs_markdown)
    docs.append(("rec_500_md",docs_rec_500_mark))
    docs_rec_500_no_mark = rec_text_splitter_500.split_documents(docs_no_markdown)
    docs.append(("rec_500_no_md", docs_rec_500_no_mark))
    docs_rec_1000_mark = rec_text_splitter_1000.split_documents(docs_markdown)
    docs.append(("rec_1000_md", docs_rec_1000_mark))
    docs_rec_1000_no_mark = rec_text_splitter_1000.split_documents(docs_no_markdown)
    docs.append(("rec_1000_no_md", docs_rec_1000_no_mark))
    md_docs = []
    for doc in docs_markdown:
        md_doc = mark_text_splitter.split_text(doc.page_content)
        for chunk in md_doc:
            chunk.metadata |= doc.metadata 
        md_docs.extend(md_doc)
    docs.append(("md_md", md_docs))

    md_docs_and_rec = []
    for doc in docs_markdown:
        md_doc_and_rec = mark_text_splitter.split_text(doc.page_content)
        for chunk in md_doc_and_rec:
            chunk.metadata |= doc.metadata 
        md_docs_and_rec.extend(md_doc_and_rec)
    md_docs_and_recs = rec_text_splitter_500.split_documents(md_docs_and_rec)
    docs.append(("md_rec_md", md_docs_and_recs))

    # Use free ollama embedding, should be changed for testing purposes TODO: Make modifiable
    embedding_nomic = OllamaEmbeddings(model="nomic-embed-text")
    embedding_snow = OllamaEmbeddings(model="snowflake-arctic-embed")

    for name, doc in docs:
         print(f"Creating {name} Vector Store now with nomic embed \n")
         vectorstore = Chroma.from_documents(
             documents=doc,
             embedding=embedding_nomic,
             persist_directory=f"chroma_nomic_db_{name}")
        
    for name, doc in docs:
        print(f"Creating {name} Vector Store now with snow embed\n")
        vectorstore = Chroma.from_documents(
            documents=doc,
            embedding=embedding_snow,
            persist_directory=f"chroma_snow_db_{name}")
    
    return 

