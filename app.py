import streamlit as st
import os
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Import our custom modules
from src.data_loader import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.question_generator import QuestionGenerator
from src.evaluation import ModelEvaluator
from src.rag_enhancement import EnhancedRetriever

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Q-Genius | AI Assessment Tool",
    page_icon="🎓",
    layout="wide"
)

# --- CACHING (SPEED BOOST) ---
@st.cache_resource
def get_vector_manager():
    return VectorStoreManager()

@st.cache_resource
def get_generator(provider, temp=0.7):
    return QuestionGenerator(model_provider=provider, temperature=temp)

@st.cache_resource
def get_enhanced_retriever():
    return EnhancedRetriever()

# --- HELPER: RESET STATE ---
def reset_app():
    st.session_state.vector_store = None
    st.session_state.questions = []
    st.session_state.processed_file_ids = []
    st.session_state.quiz_key = 0
    st.session_state.all_chunks = []

# --- HEADER ---
st.title("🎓 Q-Genius: AI-Powered Assessment Generator")
st.markdown("""
**Automated Question Generation & Answer Verification System**
*Upload multiple textbook chapters, and let Q-Genius generate a comprehensive quiz.*
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. Model Selection (REMOVED BROKEN MODELS)
    model_choice = st.radio(
        "Select AI Model:",
        (
            "Google Gemini 2.0 (Balanced)", 
            "Llama 3.3 70B (High Quality)", 
            "Llama 3.1 8B (Super Fast)"
        ),
        index=0
    )
    
    provider_map = {
        "Google Gemini 2.0 (Balanced)": "gemini",
        "Llama 3.3 70B (High Quality)": "groq",
        "Llama 3.1 8B (Super Fast)": "fast"
    }
    selected_provider = provider_map[model_choice]

    st.divider()

    # 2. File Upload
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF/TXT)", 
        type=["pdf", "txt"],
        accept_multiple_files=True,
        on_change=reset_app
    )
    
    # 3. Ablation Settings
    st.divider()
    with st.expander("🔬 Advanced Settings (Ablation Study)"):
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, step=100)
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7, step=0.1)
        retrieval_k = st.slider("Retrieval K", 3, 15, 5)
        use_reranking = st.checkbox("Use MMR Re-ranking", value=False)

    st.divider()
    
    # 4. Quiz Settings
    num_questions = st.slider("Number of Questions", 1, 10, 3)
    difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"], value="Medium")
    question_type = st.selectbox("Question Type", ["Multiple Choice", "Short Answer", "Fill in the Blank", "Long Question", "True/False"])
    
    st.divider()
    # 5. Evaluation Mode
    evaluation_mode = st.checkbox("Enable Evaluation Mode", value=False)

# --- STATE MANAGEMENT ---
if 'vector_store' not in st.session_state: st.session_state.vector_store = None
if 'questions' not in st.session_state: st.session_state.questions = []
if 'processed_file_ids' not in st.session_state: st.session_state.processed_file_ids = []
if 'quiz_key' not in st.session_state: st.session_state.quiz_key = 0
if 'all_chunks' not in st.session_state: st.session_state.all_chunks = []
if 'evaluator' not in st.session_state: st.session_state.evaluator = ModelEvaluator()
if 'ablation_results' not in st.session_state: st.session_state.ablation_results = []

# --- MAIN LOGIC ---

if uploaded_files:
    # 1. DATASET ANALYSIS
    with st.expander("📊 Dataset Analysis & Preprocessing", expanded=False):
        doc_stats = []
        for file in uploaded_files:
            # Simple check to avoid reading huge files into memory twice
            file.seek(0, os.SEEK_END)
            size_kb = file.tell() / 1024
            file.seek(0)
            doc_stats.append({'Filename': file.name, 'Size (KB)': round(size_kb, 2), 'Type': file.type})
        
        st.dataframe(pd.DataFrame(doc_stats), use_container_width=True)

    # 2. PROCESS DOCUMENTS
    current_file_ids = sorted([f.name for f in uploaded_files])
    
    if current_file_ids != st.session_state.processed_file_ids or st.session_state.vector_store is None:
        with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
            try:
                processor = DocumentProcessor(chunk_size=chunk_size)
                all_chunks = []
                
                for uploaded_file in uploaded_files:
                    uploaded_file.seek(0) # Ensure we read from start
                    chunks = processor.process_file(uploaded_file)
                    all_chunks.extend(chunks)
                
                # Update State
                st.session_state.all_chunks = all_chunks
                vs_manager = get_vector_manager()
                vector_store = vs_manager.create_vector_store(all_chunks)
                
                st.session_state.vector_store = vector_store
                st.session_state.processed_file_ids = current_file_ids
                
                # Show Chunk Visualization
                stats = processor.get_chunk_stats(all_chunks)
                with st.expander("📊 Chunk Distribution Analysis", expanded=True):
                    col1, col2 = st.columns(2)
                    col1.metric("Total Chunks", stats['total_chunks'])
                    col2.metric("Avg Size", f"{int(stats['avg_chunk_size'])} chars")
                    
                    # Graph
                    chunk_lengths = [len(c.page_content) for c in all_chunks]
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.hist(chunk_lengths, bins=20, color='skyblue', edgecolor='black')
                    ax.set_title("Chunk Size Distribution")
                    st.pyplot(fig)
                    plt.close()
                    
                st.success("✅ Knowledge Base created!")
            except Exception as e:
                st.error(f"Error processing files: {e}")

    # 3. GENERATE QUESTIONS
    if st.session_state.vector_store is not None:
        if st.button("🚀 Generate Quiz Questions"):
            st.session_state.questions = [] 
            st.session_state.quiz_key += 1
            
            with st.spinner(f"Generating with {model_choice}..."):
                try:
                    generator = get_generator(selected_provider, temp=temperature)
                    progress_bar = st.progress(0)
                    
                    # Retrieval
                    if use_reranking:
                        enhancer = get_enhanced_retriever()
                        # MMR Reranking
                        retrieved_docs = enhancer.retrieve_and_rerank(
                            st.session_state.vector_store,
                            "key concepts definitions details",
                            k=retrieval_k
                        )
                    else:
                        retrieved_docs = st.session_state.vector_store.similarity_search(
                            "key concepts definitions details", k=retrieval_k
                        )
                    
                    random.shuffle(retrieved_docs)
                    
                    for i in range(num_questions):
                        doc = retrieved_docs[i % len(retrieved_docs)]
                        context_text = doc.page_content
                        
                        start_time = time.time()
                        response = generator.generate(context_text, question_type, difficulty)
                        latency = time.time() - start_time
                        
                        st.session_state.questions.append({
                            "content": response,
                            "latency": latency,
                            "source": doc.metadata.get("source", "Unknown"),
                            "model": selected_provider
                        })
                        progress_bar.progress((i + 1) / num_questions)
                    
                    # Auto-Evaluate if enabled
                    if evaluation_mode:
                        st.session_state.evaluator.evaluate_model(selected_provider, st.session_state.questions)
                        st.session_state.evaluator.save_results()
                        
                    st.rerun()
                except Exception as e:
                    st.error(f"Generation Failed: {e}")

# --- DISPLAY ---
if st.session_state.questions:
    st.subheader(f"📝 Generated Quiz ({selected_provider})")
    
    # Summary Metrics
    total_latency = sum(q['latency'] for q in st.session_state.questions)
    avg_latency = total_latency / len(st.session_state.questions)
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Latency", f"{avg_latency:.2f}s")
    col2.metric("Total Time", f"{total_latency:.2f}s")
    col3.metric("Model", selected_provider)
    
    st.divider()
    
    for idx, q_data in enumerate(st.session_state.questions):
        full_text = q_data["content"]
        
        st.markdown(f"### Question {idx + 1}")
        st.caption(f"⚡ Latency: {q_data['latency']:.2f}s | Source: {q_data['source']}")
        
        if "|||" in full_text:
            question_part, answer_part = full_text.split("|||")
        else:
            question_part, answer_part = full_text, "Error parsing answer."

        st.markdown(question_part)
        
        unique_key = f"input_{idx}_{st.session_state.quiz_key}"
        user_ans = st.text_input(f"Your Answer:", key=unique_key)
        
        if st.button(f"Check Answer {idx+1}", key=f"btn_{idx}_{st.session_state.quiz_key}"):
            if user_ans:
                with st.spinner("Grading..."):
                    gen = get_generator(selected_provider, temp=temperature)
                    res = gen.check_answer(question_part, answer_part, user_ans)
                    if "Correct" in res and "Incorrect" not in res: st.success(res)
                    elif "Partially" in res: st.warning(res)
                    else: st.error(res)
            else:
                st.warning("Type an answer first.")

        with st.expander("Reveal Correct Answer"):
            st.info(answer_part)
        st.divider()

    # Evaluation & Export Section
    if evaluation_mode:
        with st.expander("📈 Evaluation & Comparison"):
            comparison_df = st.session_state.evaluator.compare_models()
            if not comparison_df.empty:
                st.dataframe(comparison_df)
                
                # Latency Comparison Chart
                fig, ax = plt.subplots()
                comparison_df.groupby('model')['avg_latency'].mean().plot(kind='bar', ax=ax, color='teal')
                ax.set_title("Average Latency by Model")
                st.pyplot(fig)

else:
    if not uploaded_files:
        st.info("👋 Upload documents to start.")