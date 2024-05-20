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

# client = pymongo.MongoClient("mongodb://10.0.0.10:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.4")
# db = client['clean']
# collection = db['prod_des_vn']

# data = collection.find()

# df_raw = pd.DataFrame(list(data))
# df_raw = df_raw[['description_en', 'product_name_en', 'ingredients_en']]
# rc = json.loads(df_raw.to_json(orient='records'))

def metadata_func(record: dict, metadata: dict) -> dict:
    # metadata["product_name"] = record.get("product_name_en")
    if not record.get("ingredients_en"):
        metadata["ingredients"] = ""
    else:
        metadata["ingredients"] = record.get("ingredients_en")

    if not record.get("description_en"):
        metadata["description"] = ""
    else:
        metadata["description"] = record.get("description_en")
    return metadata

loader = JSONLoader(file_path="./out.json", jq_schema='.[]', content_key = ".product_name_en", is_content_key_jq_parsable=True)
documents = loader.load()
# print(documents[-3:])

llm = Ollama(model="llama3", base_url="http://10.0.0.10:11434")
embed_func = OllamaEmbeddings(model = 'llama3', base_url="http://10.0.0.10:11434")
db = Chroma.from_documents(documents, embed_func)
retriever = db.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    query = input("Question: ")
    if query == "":
        break
    else:
        print(chain.invoke(query))