import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from utils.document_loader import DocumentLoader
from config.settings import REPOSITORY_DIR, REPOSITORY_INDEX

class RepositoryManager:
    def __init__(self):
        self.repository_dir = REPOSITORY_DIR
        self.index_file = REPOSITORY_INDEX
        self.repository_dir.mkdir(parents=True, exist_ok=True)
        self.load_index()

    def load_index(self):
        """Load repository index"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {
                'documents': {},
                'collections': {},
                'last_updated': datetime.now().isoformat()
            }
            self.save_index()

    def save_index(self):
        """Save repository index"""
        self.index['last_updated'] = datetime.now().isoformat()
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=4)

    def add_document(self, file, collection_name: str = "default") -> Dict:
        """Add document to repository"""
        # Generate unique document ID
        doc_id = f"doc_{len(self.index['documents']) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create collection if it doesn't exist
        if collection_name not in self.index['collections']:
            self.index['collections'][collection_name] = {
                'created_at': datetime.now().isoformat(),
                'documents': []
            }

        # Save document to repository
        doc_path = self.repository_dir / collection_name / file.name
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(doc_path, 'wb') as f:
            f.write(file.getvalue())

        # Add to index
        doc_info = {
            'id': doc_id,
            'filename': file.name,
            'path': str(doc_path),
            'collection': collection_name,
            'added_at': datetime.now().isoformat(),
            'file_type': Path(file.name).suffix.lower(),
            'file_size': os.path.getsize(doc_path)
        }
        
        self.index['documents'][doc_id] = doc_info
        self.index['collections'][collection_name]['documents'].append(doc_id)
        
        self.save_index()
        return doc_info

    def get_document(self, doc_id: str) -> Optional[Path]:
        """Get document path by ID"""
        if doc_id in self.index['documents']:
            path = Path(self.index['documents'][doc_id]['path'])
            if path.exists():
                return path
        return None

    def get_collection_documents(self, collection_name: str) -> List[Dict]:
        """Get all documents in a collection"""
        if collection_name in self.index['collections']:
            docs = []
            for doc_id in self.index['collections'][collection_name]['documents']:
                if doc_id in self.index['documents']:
                    docs.append(self.index['documents'][doc_id])
            return docs
        return []

    def load_collection_documents(self, collection_name: str) -> List:
        """Load all documents in a collection"""
        documents = []
        for doc_info in self.get_collection_documents(collection_name):
            doc_path = Path(doc_info['path'])
            if doc_path.exists():
                try:
                    with open(doc_path, 'rb') as f:
                        doc = DocumentLoader.load_document_from_file(doc_path)
                        documents.extend(doc)
                except Exception as e:
                    print(f"Error loading document {doc_path}: {e}")
        return documents

    def get_collections(self) -> List[str]:
        """Get list of all collections"""
        return list(self.index['collections'].keys())

    def get_repository_stats(self) -> Dict:
        """Get repository statistics"""
        return {
            'total_documents': len(self.index['documents']),
            'total_collections': len(self.index['collections']),
            'total_size': sum(doc['file_size'] for doc in self.index['documents'].values()),
            'last_updated': self.index['last_updated']
        }

    def search_documents(self, query: str) -> List[Dict]:
        """Search documents by filename or content"""
        query = query.lower()
        results = []
        for doc_id, doc_info in self.index['documents'].items():
            if query in doc_info['filename'].lower():
                results.append(doc_info)
        return results

    def get_document_info_df(self) -> pd.DataFrame:
        """Get document information as DataFrame"""
        if not self.index['documents']:
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(self.index['documents'], orient='index')
        df['added_at'] = pd.to_datetime(df['added_at'])
        df['file_size_kb'] = df['file_size'].apply(lambda x: f"{x/1024:.1f}")
        return df

    def remove_document(self, doc_id: str) -> bool:
        """Remove document from repository"""
        if doc_id in self.index['documents']:
            doc_info = self.index['documents'][doc_id]
            # Remove file
            doc_path = Path(doc_info['path'])
            if doc_path.exists():
                doc_path.unlink()
            
            # Remove from collection
            collection = doc_info['collection']
            if collection in self.index['collections']:
                self.index['collections'][collection]['documents'].remove(doc_id)
            
            # Remove from index
            del self.index['documents'][doc_id]
            self.save_index()
            return True
        return False

    def clear_collection(self, collection_name: str) -> bool:
        """Clear all documents in a collection"""
        if collection_name in self.index['collections']:
            for doc_id in self.index['collections'][collection_name]['documents']:
                self.remove_document(doc_id)
            return True
        return False