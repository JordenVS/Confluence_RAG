from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import createRetriever

model = OllamaLLM(model = "llama3.2")

template = """
You are an expert in answering questions about the Europeana Knowledge base

Here are some relevant context chunks: {context}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

result = chain.invoke({"context": [], "question": "What is Europeana?"})

retriever = createRetriever()


print(retriever.invoke("The Netherlands"))
print(result)