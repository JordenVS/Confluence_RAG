from langchain_ollama.llms import OllamaLLM
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from vector import createRetriever

embedding = OllamaEmbeddings(model="nomic-embed-text")

model = OllamaLLM(model = "llama3.2")

#retriever = createRetriever()
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever)

response = qa_chain.run("What are the 3 steps when enriching data with organisation entities?")
print(response)
