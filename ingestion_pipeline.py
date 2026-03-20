import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path = "docs"):
    print(f"Loading documents from {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory {docs_path} does not exist")


    loader=DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}

    )

    documents=loader.load()

    if len(documents) == 0:
        raise ValueError(f"No documents found in {docs_path}")


    
    # for i,doc in enumerate(documents[0:]):
    #     print(f"\nDocument {i+1}:")
    #     print(f"source :{doc.metadata['source']}")
    #     print(f"content len  :{len(doc.page_content)} characters")
    #     print(f"content :{doc.page_content[:100]}...")
    #     print(f"metadata :{doc.metadata}")
    
    return documents



def split_documents(documents,chunk_size=500,chunk_overlap=0):
    print(f"Splitting documents into chunks of {chunk_size}")
    
    text_splitter=CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks=text_splitter.split_documents(documents)
    
    # for i,chunk in enumerate(chunks[0:3]):
    #     print(f"\n________Chunk {i+1}________")
    #     print(f"source:{chunk.metadata['source']}")
    #     print(f"chunk len:{len(chunk.page_content)} characters")
    #     print("----content  Below----")
    #     print(chunk.page_content)

    # if len(chunks) >3:
    #     print(f"\n\n...{len(chunks)-3} more chunks not shown...")
        
    return chunks

def create_vector_store(chunks,persist_dir="db\chroma_db"):
    print(f"Creating vector store in {persist_dir}")

    embedding_model=HuggingFaceBgeEmbeddings(model_name = "BAAI/bge-small-en")

    vector_store=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    print(f"Vector store created successfully in {persist_dir}")
    return vector_store

        

def main():
    print("Starting ingestion pipeline...")

    documents = load_documents(docs_path="docs")

    chunks=split_documents(documents)

    vector_store=create_vector_store(chunks)


    

if __name__ == "__main__":
    main()