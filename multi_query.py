from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import json
import ast


from urllib3 import response

persistent_directory = "db/chroma_db"
llm = Ollama(model="mistral")
embedding_model = HuggingFaceBgeEmbeddings(model_name ="BAAI/bge-small-en")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# class queryvariations(BaseModel):
#     queries : List[str]



original_query = "what is the drawback of llm?"
print(f"Original Query: {original_query}\n")




prompt = f"""
Generate exactly 3 query variations.

return 3 alternative queries that rephrase or approach the same question from every angles.

STRICT RULES:
- Output must be a valid Python list
- Each item must be a string
- No numbering
- No explanation
- No extra text

Output format ONLY:
["query1", "query2", "query3"]

Original query: {original_query}
"""

# prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

# Original query: {original_query}

# Return 3 alternative queries that rephrase or approach the same question from every angles.
# Return ONLY in this JSON format:
# {{
#   "queries": ["...", "...", "..."]
# }}
# """

response = llm.invoke(prompt)

query_variations = ast.literal_eval(response)




print("Generated Query Variations:")
for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

print("\n" + "="*60)

