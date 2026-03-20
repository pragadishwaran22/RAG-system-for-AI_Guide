from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core import embeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

load_dotenv()


embeddings = HuggingFaceBgeEmbeddings(model_name ="BAAI/bge-small-en")
persist_dir = "db/chroma_db"
model = Ollama(model = "mistral")

db = Chroma(persist_directory = persist_dir,embedding_function = embeddings)

chat_history = []

def ask_question(user_question):
    print(f"\n---you asked : {user_question}")

    if chat_history:
        messages =[
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question.")
        ] + chat_history +[
            HumanMessage(content =f"new question:{user_question}")
        ]
        
        result = model.invoke(messages)
        search_question =result.strip()
        print(f"searching for {search_question}")
    else:
        search_question = user_question

    retriver = db.as_retriever(search_kwargs = {"k":3})
    relavant_docs = retriver.invoke(search_question)

    print(f"found {len(relavant_docs)} relavant_documents")

    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in relavant_docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result

    chat_history.append(HumanMessage(content = user_question))
    chat_history.append(AIMessage(content = answer))

    print(f"Answer generated :{answer}")
    return answer

    




def start_chat():
    question = input("ask your question :")
    ask_question(question)
    verify = input("still do you have any question (yes/no) ? :")
    if verify.lower()=="yes":
        start_chat()
    else:
        print("Good Bye :) Feel Free to Ask Anything")


if __name__ == "__main__":
    start_chat()



