from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from vector import createRetriever, createDBs
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
import pandas as pd
import os
import csv

def testQueriesLLM(qa_chain):
    correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model="ollama:llama3.2",
)
    df = pd.read_csv(filepath_or_buffer="queries.csv", sep=";")
    inputs = []
    outputs = []
    ref_outputs = []
    for i in range(len(df['question'])):
        response = qa_chain.invoke(df['question'][i])
        print(response.get("result"))
        inputs.append(df['question'][i])
        outputs.append(response.get("result"))
        print("\nExpected:\n")
        print(df['expected_answer'][i])
        ref_outputs.append(df['expected_answer'][i])
        
    eval_result = correctness_evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=ref_outputs
    )
    
    print(eval_result)

def testQueriesCSV(qa_chain, dir):
    df = pd.read_csv(filepath_or_buffer="queries.csv", sep=";")
    inputs = []
    outputs = []
    ref_outputs = []
    for i in range(len(df['question'])):
        response = qa_chain.invoke(df['question'][i])
        inputs.append(df['question'][i])
        outputs.append(response.get("result"))
        ref_outputs.append(df['expected_answer'][i])

    # Define the output file
    os.makedirs("results", exist_ok=True)
    csv_filename = f"results/{dir}.csv"

    # Write header and data row (append mode ensures you can keep adding more rows)
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write header only if the file is empty
        if file.tell() == 0:
            writer.writerow(["Input Question", "LLM Output", "Reference Output"])

        for question, llm_output, reference_output in zip(inputs, outputs, ref_outputs):
            writer.writerow([question, llm_output, reference_output])



#embedding = OllamaEmbeddings(model="nomic-embed-text")
#embedding_snow = OllamaEmbeddings(model="snowflake-arctic-embed")

#model = OllamaLLM(model = "llama3.2")

#dir = "chroma_nomic_db_rec_500_md"
#dir = "chroma_nomic_db_rec_500_no_md"
#dir = "chroma_nomic_db_rec_1000_md"
#dir = "chroma_nomic_db_rec_1000_no_md"
#dir = "chroma_snow_db_rec_500_md"
#dir = "chroma_snow_db_rec_500_no_md"
#vectorstore = Chroma(persist_directory=dir, embedding_function=embedding_snow)
#retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":6})

#qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever)

#testQueriesLLM(qa_chain)
#testQueriesCSV(qa_chain, dir)
createDBs()
