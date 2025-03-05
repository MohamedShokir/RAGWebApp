import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import BinaryIO
from config.settings import UPLOAD_DIRECTORY, METADATA_FILE, SUPPORTED_FORMATS

class DocumentManager:
    def __init__(self):
        self.metadata_file = METADATA_FILE
        self.load_metadata()

    def load_metadata(self):
        """Load document metadata from JSON file"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def save_metadata(self):
        """Save document metadata to JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)

    def add_document(self, file: BinaryIO, embedding_model: str) -> str:
        """Add document to storage with metadata"""
        file_hash = self._calculate_file_hash(file)
        
        # Save file
        filename = Path(file.name)
        file_path = Path(UPLOAD_DIRECTORY) / filename
        with open(file_path, 'wb') as f:
            f.write(file.getbuffer())

        # Store metadata
        self.metadata[file_hash] = {
            'filename': file.name,
            'upload_time': datetime.now().isoformat(),
            'embedding_model': embedding_model,
            'file_size': os.path.getsize(file_path),
            'file_type': filename.suffix,
            'path': str(file_path)
        }
        self.save_metadata()
        return file_hash

    def _calculate_file_hash(self, file: BinaryIO) -> str:
        """Calculate SHA-256 hash of file content"""
        file_content = file.getvalue()
        return hashlib.sha256(file_content).hexdigest()

    def get_document_info(self) -> pd.DataFrame:
        """Get information about all stored documents"""
        return pd.DataFrame.from_dict(self.metadata, orient='index')

    def remove_document(self, file_hash: str) -> bool:
        """Remove document and its metadata"""
        if file_hash in self.metadata:
            file_path = Path(self.metadata[file_hash]['path'])
            if file_path.exists():
                file_path.unlink()
            del self.metadata[file_hash]
            self.save_metadata()
            return True
        return False