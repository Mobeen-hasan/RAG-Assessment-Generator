# Q-Genius: AI-Powered Assessment Generator

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Library-green?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Google%20Gemini-AI-blue?style=for-the-badge)

**Q-Genius** is a state-of-the-art Retrieval-Augmented Generation (RAG) platform designed to transform static documents (PDFs/TXTs) into interactive, high-quality assessments. Leveraging advanced LLMs like Google Gemini 2.0 and Llama 3.3, it automates the creation of quizzes, evaluates student responses, and provides detailed performance analytics.

## 🚀 Key Features

- **Multi-Document RAG**: Process and index multiple large-scale documents simultaneously using optimized chunking strategies.
- **Intelligent Question Generation**: Generate Multiple Choice, Short Answer, True/False, and Long-form questions across varied difficulty levels.
- **Model Comparison & Evaluation**: Built-in ablation study tools to compare latency and quality between Gemini, Llama 3.3 70B, and Llama 3.1 8B.
- **Advanced Retrieval**: Utilizes MMR (Maximum Marginal Relevance) re-ranking and vector similarity search for high-precision context retrieval.
- **Real-time Grading**: Automated answer verification system that provides instant feedback and grading for user inputs.
- **Dataset Analysis**: Visualizations of chunk distributions, token counts, and retrieval metrics.

## 🛠️ Tech Stack

- **Framework**: [Streamlit](https://streamlit.io/)
- **Orchestration**: LangChain
- **Models**: Google Gemini 2.0, Groq (Llama 3.3/3.1)
- **Vector DB**: FAISS / VectorStoreManager
- **Data Processing**: PyPDF2, Pandas, Matplotlib

## 🔬 Ablation & Advanced Settings

Q-Genius allows fine-tuning of the RAG pipeline:
- **Chunk Size Tuning**: Adjust context window sizes for better relevance.
- **Temperature Control**: Manage the creativity vs. precision of the generated assessment.
- **MMR Re-ranking**: Optimize retrieval diversity to avoid redundant questions.

## 📥 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mobeen-hasan/RAG-Assessment-Generator.git
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up Environment Variables**:
   Create a `.env` file with your `GOOGLE_API_KEY` and `GROQ_API_KEY`.
4. **Launch the app**:
   ```bash
   streamlit run app.py
   ```

