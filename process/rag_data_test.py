from datasets import load_dataset
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
import sys
import os
import pandas as pd
import chromadb

dataset = load_dataset("neural-bridge/rag-dataset-12000", split="train")
dataset = pd.DataFrame(dataset)

chroma_client = chromadb.Client()
corpus_collection = chroma_client.create_collection(name='rag_data', embedding_function=OllamaEmbeddings(model="phi3:latest"))

batch_size = 10
# with SuppressStdout():
for i in range(0, len(dataset), batch_size):
    batch_df = dataset[i:i+batch_size]
    print(len(("Context: " + batch_df['context'] + '. ' + "Question: " + batch_df['question'] + '. ' + "Answer: " + batch_df['answer']).tolist()))
    print(len(list(map(lambda x: str(x + i), range(batch_size)))))
    corpus_collection.add(
        ids= list(map(lambda x: str(x + i), range(batch_size))),
        documents= ("Context: " + batch_df['context'] + '. ' + "Question: " + batch_df['question'] + '. ' + "Answer: " + batch_df['answer']).tolist(),
        metadatas=[]
    )

while True:
    query = input("\nQuery: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue

    # Prompt
    template = """
    Context: {context}
    Question: {question}
    Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    llm = Ollama(model="phi3:latest", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=corpus_collection.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    result = qa_chain({"query": query})