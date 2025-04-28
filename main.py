from langchain_ollama.llms import OllamaLLM
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from vector import createRetriever

embedding = OllamaEmbeddings(model="nomic-embed-text")

model = OllamaLLM(model = "llama3.2")

#retriever = createRetriever()
vectorstore = Chroma(persist_directory="chroma_db_txt", embedding_function=embedding)
retriever = vectorstore.as_retriever()
# chroma_db is the database created by setting markdown_format to true, chunking with RecursiveCharacterTextSplitter, chunk size 500, overlap 100 with separators["\n\n", "\n", " ", ""] and nomic-embed-text as embedding.
# chroma_db_text is the database created by setting markdown_format to false, chunking with RecursiveCharacterTextSplitter, chunk size 500, overlap 100 and nomic-embed-text as embedding.

qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever)

response = qa_chain.run("What did the dutch state regarding article 14?")
print(response)
