from typing import List, Any
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredPowerPointLoader
)
from pathlib import Path
import tempfile

class DocumentLoader:
    """Handles document loading for different file types"""
    
    LOADER_MAPPING = {
        '.txt': TextLoader,
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.csv': CSVLoader,
        '.pptx': UnstructuredPowerPointLoader
    }

    @staticmethod
    def load_document(file, file_path: str) -> List[Any]:
        """Load document using appropriate loader based on file extension"""
        ext = Path(file.name).suffix.lower()
        
        if ext not in DocumentLoader.LOADER_MAPPING:
            raise ValueError(f"Unsupported file type: {ext}")
            
        # Create a temporary file to handle the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file.flush()
            
            # Load the document using appropriate loader
            loader_class = DocumentLoader.LOADER_MAPPING[ext]
            loader = loader_class(tmp_file.name)
            documents = loader.load()
            
            # Clean up temporary file
            Path(tmp_file.name).unlink()
            
            return documents

    @staticmethod
    def process_uploaded_files(files: List[Any]) -> List[Any]:
        """Process multiple uploaded files"""
        all_documents = []
        
        for file in files:
            try:
                documents = DocumentLoader.load_document(file, file.name)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error processing {file.name}: {str(e)}")
                continue
                
        return all_documents
    
    @staticmethod
    def load_document_from_file(file_path: Path) -> List[Any]:
        """Load document from file path"""
        ext = file_path.suffix.lower()
        
        if ext not in DocumentLoader.LOADER_MAPPING:
            raise ValueError(f"Unsupported file type: {ext}")
            
        loader_class = DocumentLoader.LOADER_MAPPING[ext]
        loader = loader_class(str(file_path))
        return loader.load()