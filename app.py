import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import time
from utils.document_manager import DocumentManager
from utils.rag_optimizer import RAGOptimizer
from utils.performance_monitor import PerformanceMonitor
from utils.model_manager import ModelManager
from utils.document_loader import DocumentLoader
from utils.repository_manager import RepositoryManager
from config.settings import EMBEDDING_MODELS, SUPPORTED_FORMATS
import pandas as pd

# Initialize components
doc_manager = DocumentManager()
model_manager = ModelManager()
performance_monitor = PerformanceMonitor()
repo_manager = RepositoryManager()

def setup_qa_chain(vectorstore, model_name: str):
    """Set up the question-answering chain"""
    llm = Ollama(model=model_name)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )
    )

def display_repository_ui(unique_id):
    """Display document repository interface"""
    st.sidebar.header("Document Repository")
    
    # Display collections
    collections = repo_manager.get_collections()
    if collections:
        selected_collection = st.sidebar.selectbox(
            "Select Collection",
            collections,
            key=f"repository_collection_select_{unique_id}"  # Ensure unique key
        )
        
        # Display documents in collection
        docs = repo_manager.get_collection_documents(selected_collection)
        if docs:
            st.sidebar.subheader("Documents in Collection")
            df = pd.DataFrame(docs)
            df['added_at'] = pd.to_datetime(df['added_at']).dt.strftime('%Y-%m-%d %H:%M')
            df['file_size'] = df['file_size'].apply(lambda x: f"{x/1024:.1f} KB")
            
            st.sidebar.dataframe(
                df[['filename', 'added_at', 'file_size']],
                hide_index=True,
                use_container_width=True
            )
            
            # Option to remove documents
            if st.sidebar.button("Clear Collection", key=f"clear_collection_btn_{unique_id}"):  # Ensure unique key
                if repo_manager.clear_collection(selected_collection):
                    st.sidebar.success(f"Cleared collection: {selected_collection}")
                    st.experimental_rerun()

def main():
    st.title("Enhanced RAG System with Document Repository")

    # Check Ollama status
    st.sidebar.header("System Status")
    ollama_status = st.sidebar.empty()

    if not model_manager.check_ollama_status():
        ollama_status.error("❌ Ollama is not running")
        st.stop()
    ollama_status.success("✅ Ollama is running")

    # Model selection
    installed_models = model_manager.get_installed_models()
    if not installed_models:
        st.warning("No Ollama models found. Please install models using 'ollama pull model-name'")
        st.stop()

    llm_model = st.sidebar.selectbox(
        "Select LLM Model",
        installed_models,
        key="llm_model_select"  # Ensure unique key
    )

    embedding_model = st.sidebar.selectbox(
        "Select Embedding Model",
        list(EMBEDDING_MODELS.keys()),
        key="embedding_model_select"  # Ensure unique key
    )

    # Initialize RAG optimizer
    rag = RAGOptimizer(llm_model, EMBEDDING_MODELS[embedding_model])

    # Collection selection/creation
    collection_name = st.text_input(
        "Collection Name", 
        "default",
        key="collection_name_input"  # Ensure unique key
    )

    # File upload
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=list(SUPPORTED_FORMATS.keys()),
        accept_multiple_files=True,
        key="document_uploader"  # Ensure unique key
    )

    if uploaded_files:
        try:
            with st.spinner("Processing documents..."):
                # Add documents to repository
                for file in uploaded_files:
                    repo_manager.add_document(file, collection_name)
                
                # Load all documents from the collection
                documents = repo_manager.load_collection_documents(collection_name)
                
                # Process documents with RAG optimizer
                chunks = rag.create_chunks(documents)
                vectorstore = rag.setup_vectorstore(chunks)
                
            st.success("Documents processed and added to repository!")
            
            # Display repository UI
            display_repository_ui(unique_id="main")  # Pass unique ID
            
            # Question-answering section
            st.header("Ask Questions")
            question = st.text_input(
                "Enter your question:",
                key="question_input"  # Ensure unique key
            )
            
            if question:
                with st.spinner("Generating answer..."):
                    try:
                        processed_question = rag.process_text(question)
                        qa_chain = setup_qa_chain(vectorstore, llm_model)
                        answer = qa_chain.run(processed_question)
                        
                        st.subheader("Answer:")
                        st.write(answer)
                        
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

    # Display repository even if no new files are uploaded
    display_repository_ui(unique_id="sidebar")  # Pass unique ID

    # Display system metrics
    with st.sidebar.expander("System Metrics"):
        metrics = performance_monitor.get_system_metrics()
        st.write("Memory Usage:", f"{metrics['memory_used_percent']}%")
        st.write("Available Memory:", f"{metrics['memory_available_gb']:.1f} GB")
        st.write("CPU Usage:", f"{metrics['cpu_percent']}%")

    # Display optimization tips
    with st.sidebar.expander("RAG Optimization Tips"):
        st.markdown("""
        ### Tips for Better Results:
        1. **Document Quality**:
           - Use clear, well-formatted documents
           - Ensure text is machine-readable
           - Split large documents appropriately
        
        2. **Question Format**:
           - Be specific and clear
           - Use complete sentences
           - Avoid complex queries initially
        
        3. **System Performance**:
           - Monitor response times
           - Clear database periodically
           - Update models regularly
        """)

if __name__ == "__main__":
    main()