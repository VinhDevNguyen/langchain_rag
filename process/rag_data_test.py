from datasets import load_dataset
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings.ollama import OllamaEmbeddings
import pandas as pd
from langchain.chains.combine_documents import create_stuff_documents_chain
import chromadb

dataset = load_dataset("neural-bridge/rag-dataset-12000", split="train")
dataset = pd.DataFrame(dataset)[:50]

class DefChromaEF(OllamaEmbeddings):
  def __init__(self,ef):
    self.ef = ef
    self.model = "phi3"

  def embed_documents(self,texts):
    return self.ef(texts)

  def embed_query(self, query):
    return self.ef([query])[0]

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name='rag_data', embedding_function=DefChromaEF.embed_documents)

llm = Ollama(model="phi3")

batch_size = 10

for i in range(0, len(dataset), batch_size):
    batch_df = dataset[i:i+batch_size]
    collection.add(
        ids= list(map(lambda x: str(x + i), range(batch_size))),
        documents= ("Context: " + batch_df['context'] + '. ' + "Question: " + batch_df['question'] + '. ' + "Answer: " + batch_df['answer']).tolist(),
    )

# Prompt
prompt = ChatPromptTemplate.from_template(
    """
            Answer the following question only based on the given context
                                                    
            <context>
            {context}
            </context>
                                                    
            Question: {input}
"""
)

db = Chroma(client = chroma_client, collection_name="rag_data", embedding_function=DefChromaEF.embed_documents)

## Retrieve context from vector store
docs_chain = create_stuff_documents_chain(llm, prompt)
retriever = db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, docs_chain)

response = retrieval_chain.invoke({"input": "What are the unique features of the Coolands for Twitter app?"})