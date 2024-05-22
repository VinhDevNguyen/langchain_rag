import pandas as pd
import pymongo
import json
from langchain_community.document_loaders import JSONLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class ChromaDatabase:
    def __init__(self, model, embed, url, db_dir):
        self.llm = Ollama(model = model, base_url = url)
        self.embed_func = OllamaEmbeddings(model = embed, base_url = url)
        self.db_dir = db_dir


    def embed(self, documents, collection_name = 'test'):
        db = Chroma.from_documents(documents, self.embed_func, persist_directory=self.db_dir, collection_name=collection_name)
        
        return db

    def load(self, collection_name):
        db = Chroma(collection_name= collection_name, persist_directory=self.db_dir, embedding_function=self.embed_func)

        return db

    def build_chain(self, db):
        retriever = db.as_retriever()

        template = """You are an e-commerce product consultant. Recommend three to five products based on this context:
        {context}. 
        List ingredients if available with their product code(s), only use data from the context.

        Answer the following question:
        {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
            )
    
        return chain

    def query(self, chain, query):
        print(chain.invoke(query))