from langchain_chroma import Chroma
from langchain_chroma.vectorstores import cosine_similarity
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_dir = "db/chroma_db"
embeddings_model = HuggingFaceBgeEmbeddings(model_name ="BAAI/bge-small-en")

db = Chroma(
    persist_directory = persistent_dir,
    embedding_function = embeddings_model,
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
print(f"User Query is :{query}")
print("_______context_______")

for i,docs in enumerate(relevant_docs,1):
    print(f"document {i} :\n{docs.page_content}\n")


    