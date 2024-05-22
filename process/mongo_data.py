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
# df_raw = df_raw[['description_en', 'product_name_en', 'ingredients_en', "product_code"]]
# rc = json.loads(df_raw.to_json(orient='records'))

# with open("out.json", 'w') as file:
#     json.dump(rc, file, indent=4)  # `indent=4` for pretty-printing

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["product_name"] = record.get("product_name_en")
    if not record.get("ingredients_en"):
        metadata["ingredients"] = ""
    else:
        metadata["ingredients"] = record.get("ingredients_en")
    
    metadata['product_code'] = record.get('product_code')
    return metadata

loader = JSONLoader(file_path="./out.json", jq_schema='.[]', content_key="description_en",text_content=False, metadata_func=metadata_func)
documents = loader.load()
# print(documents[-10:])

llm = Ollama(model="phi3", base_url="http://172.17.0.1:11434")
embed_func = OllamaEmbeddings(model = 'nomic-embed-text', base_url="http://172.17.0.1:11434")
# db = Chroma.from_documents(documents, embed_func, persist_directory="./chroma_db", collection_name="test_nomic")
db = Chroma(collection_name='test_nomic', persist_directory="./chroma_db", embedding_function=embed_func)
retriever = db.as_retriever()

# query_prompt = "List products for damaged hair"

# # Perform the search
# results = db.similarity_search(query_prompt)

# # Display the results
# for result in results:
#     print(result.page_content)

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
    | llm
    | StrOutputParser()
)

while True:
    query = input("Question: ")
    if query == "":
        break
    else:
        print(chain.invoke(query))