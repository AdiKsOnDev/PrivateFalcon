import os
import glob
from dotenv import load_dotenv
from multiprocessing import Pool
from typing import List
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader
)

if not load_dotenv():
    print("Could not load .env file or it is empty.")
    exit(1)

DB_DIRECTORY = os.environ.get('DB_DIRECTORY')
CHUNK_SIZE = os.environ.get('CHUNK_SIZE')
CHUNK_OVERLAP = os.environ.get('CHUNK_OVERLAP')

if DB_DIRECTORY is None:
    raise Exception("Set the DB_DIRECTORY in the '.env' file!")

# Chroma settings
CHROMA_SETTINGS = Settings(
        persist_directory=DB_DIRECTORY,
        anonymized_telemetry=False # Disable usage information collecting
)

# Map file extensions to document loaders
LOADERS = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"})
}

def load_document(file):
    """ Loads a single document

        Arguments: 
            file (str) --> The path to a file
    """
    extension = f'.{file.split(".")[-1].lower()}'
    
    if extension not in LOADERS:
        raise Exception(f"Files with the extension {extension} are not supported")
    
    loader_class, arguments = LOADERS[extension]
    
    loader = loader_class(file, **arguments)
    return loader.load()

def load_directory(dir, loaded_files):
    """ Loads the documents from a certain directory
        Uses the `load_document()` function

        Arguments: 
            dir (str) --> The path to a directory
            loaded_files (List[str]) --> Files that are already loaded and should be ignored
    """
    files = []
    for extension in LOADERS:
        files.extend(
            glob.glob(os.path.join(dir, f"**/*{extension.lower()}"), recursive=True)
        )

    ignored_files = [path for path in files if path not in loaded_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        for i, docs in enumerate(pool.imap_unordered(load_document, ignored_files)):
            results.extend(docs)
    
    return results

def split_documents(loaded_files):
    """ Process the documents and split them into chunks

        Arguments:
            loaded_files (List[str]) --> Files that are already loaded and should be ignored
    """
    print("--> Loading the Documents <--")
    
    documents = load_directory(DB_DIRECTORY, loaded_files)

    if not documents:
        raise Exception("No new documents to ingest!")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    split_documents = splitter.split_documents(documents)

    print(f"Split the documents into {len(documents)} chunks (Max. {CHUNK_SIZE} tokens each)")

    return split_documents