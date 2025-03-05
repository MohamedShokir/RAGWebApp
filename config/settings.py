import os
from pathlib import Path
from chromadb.config import Settings

# Directory Configuration
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIRECTORY = BASE_DIR / "uploaded_documents"
PERSIST_DIRECTORY = BASE_DIR / "db"
METADATA_FILE = BASE_DIR / "document_metadata.json"

# Create necessary directories
UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Chroma Settings
CHROMA_SETTINGS = Settings(
    persist_directory=str(PERSIST_DIRECTORY),
    anonymized_telemetry=False
)

# Supported File Types
SUPPORTED_FORMATS = {
    ".txt": "Text",
    ".pdf": "PDF",
    ".docx": "Word",
    ".csv": "CSV",
    ".pptx": "PowerPoint"
}


REPOSITORY_DIR = BASE_DIR / "document_repository"
REPOSITORY_INDEX = REPOSITORY_DIR / "repository_index.json"

# Create repository directory
REPOSITORY_DIR.mkdir(parents=True, exist_ok=True)

# Model Settings
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
}

# Document Processing Settings
CHUNK_SETTINGS = {
    "mistral": {"size": 2000, "overlap": 200},
    "mixtral": {"size": 3000, "overlap": 300},
    "llama2": {"size": 1000, "overlap": 100},
    "default": {"size": 1000, "overlap": 100}
}