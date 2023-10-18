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
    UnstructuredEmailLoader,
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