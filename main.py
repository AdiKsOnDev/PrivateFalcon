import os
import chromadb
import time
import torch
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from chromadb.config import Settings
from transformers import AutoTokenizer, pipeline

if not load_dotenv():
    print("Could not load .env file or it is empty.")
    exit(1)
    
DB_DIRECTORY = os.environ.get('DB_DIRECTORY')
EMBEDDINGS_MODEL = os.environ.get('EMBEDDINGS_MODEL')
SOURCE_CHUNKS = int(os.environ.get('SOURCE_CHUNKS', 4))

# Chroma settings
CHROMA_SETTINGS = Settings(
    persist_directory=DB_DIRECTORY,
    anonymized_telemetry=False # Disable usage information collecting
)

def main():
    print("Entering Main")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=DB_DIRECTORY)

    model_id = "tiiuae/falcon-7b"

    data = Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = data.as_retriever(search_kwargs={"k": SOURCE_CHUNKS})
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Retriever")

    pipe = pipeline(
        model=model_id,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_new_tokens=200
        )
    llm = HuggingFacePipeline(pipeline=pipe)

    print("Made a pipeline")
    
    retrieval = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    print("Entering loop")
    while True:
        
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = retrieval(query)
        answer= res['result']
        
        end = time.time()

        # Print the result
        print("\n\n--> Question:")
        print(query)

        print(f"\n--> Answer (took {round(end - start, 2)} s.):")
        print(answer)

if __name__ == "__main__":
    main()