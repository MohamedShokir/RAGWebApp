import re
from typing import List, Dict, Any
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from config.settings import CHUNK_SETTINGS, CHROMA_SETTINGS, PERSIST_DIRECTORY

class RAGOptimizer:
    def __init__(self, model_name: str, embedding_model: str):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.chunk_settings = self._get_chunk_settings()

    def _get_chunk_settings(self) -> Dict[str, int]:
        """Get optimal chunk settings based on model"""
        return CHUNK_SETTINGS.get(self.model_name, CHUNK_SETTINGS['default'])

    def process_text(self, text: str) -> str:
        """Preprocess text for better RAG performance"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        
        # Normalize apostrophes
        text = re.sub(r'[''`]', "'", text)
        
        # Remove special characters but keep essential punctuation
        text = re.sub(r'[^\w\s.,!?;:()"\'-]', '', text)
        
        return text.strip()

    def create_chunks(self, documents: List[Any]) -> List[Any]:
        """Create optimized chunks from documents"""
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_settings['size'],
            chunk_overlap=self.chunk_settings['overlap'],
            separator="\n"
        )
        return text_splitter.split_documents(documents)

    def setup_vectorstore(self, documents: List[Any]) -> Chroma:
        """Setup vector store with optimized settings"""
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'}
        )

        # Create new vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(PERSIST_DIRECTORY)
        )
        
        # Persist the vector store
        vectorstore.persist()
        return vectorstore

    def get_existing_vectorstore(self) -> Chroma:
        """Get existing vector store if it exists"""
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        return Chroma(
            persist_directory=str(PERSIST_DIRECTORY),
            embedding_function=embeddings
        )

    def update_vectorstore(self, vectorstore: Chroma, new_documents: List[Any]) -> Chroma:
        """Update existing vector store with new documents"""
        vectorstore.add_documents(new_documents)
        vectorstore.persist()
        return vectorstore