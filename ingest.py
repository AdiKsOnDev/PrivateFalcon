import os
import glob
from dotenv import load_dotenv
from multiprocessing import Pool
from typing import List
from chromadb.config import Settings

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
            file (string) --> The path to a file
    """
    extension = f'.{file.split(".")[-1].lower()}'
    
    if extension not in LOADERS:
        raise Exception(f"Files with the extension {extension} are not supported")
    
    loader_class, arguments = LOADERS[extension]
    
    loader = loader_class(file, **arguments)
    return loader.load()

def load_directory(dir):
    """ Loads the documents from a certain directory
        Uses the `load_document()` function

        Arguments: 
            dir (string) --> The path to a directory
    """
    files = []
    for extension in LOADERS:
        files.extend(
            glob.glob(os.path.join(dir, f"**/*{extension.lower()}"), recursive=True)
        )

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        for i, docs in enumerate(pool.imap_unordered(load_document)):
            results.extend(docs)
    
    return results