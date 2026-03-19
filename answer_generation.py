from langchain_chroma import Chroma
from langchain_chroma.vectorstores import cosine_similarity
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

model = Ollama(model = "mistral")
embeddings = HuggingFaceBgeEmbeddings(model_name ="BAAI/bge-small-en")
persist_dir = "db/chroma_db"

db = Chroma(
    embedding_function = embeddings,
    persist_directory= persist_dir,
    collection_metadata = {"hnsw:space":"cosine"}
)

query =input( "ask your question:")


retriver = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {
        "k":3,
        "score_threshold":0.5
    }
)

relevant_docs = retriver.invoke(query)


model = Ollama(model = "mistral")

combined_input = f"""Based on the following documents only you have to answer the {query}

Documents:{chr(10).join([f"-{doc.page_content}"for doc in relevant_docs])}

please provide  a clear, helpfull answer using only the infromation from these documents. if you can't find the answer in the documents, say "I dont have enough information to answer the question based on the provide documents"
"""

messages = [
    SystemMessage(content="you are a helpfull assistant"),
    HumanMessage(content = combined_input),
]

results = model.invoke(messages)

print("\n_______Generated Response________")


print("content only")
print(results)

