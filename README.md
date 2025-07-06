# Agentic-RAG
# Biology Tutor Chatbot (RAG + LangGraph + LLaMA 3)

An academic-grade intelligent chatbot that answers **12th-grade NCERT Biology** questions with textbook + web knowledge using **LangGraph, LangChain, FAISS, and Meta LLaMA 3.2 1B Instruct**.

---

## 📌 Project Overview

This notebook-based project builds a **Retrieval-Augmented Generation (RAG)** system with the following features:

- 🔍 **Vector DB Search (FAISS)** over NCERT Biology PDF (`lebo106.pdf`)
- 🌐 **Live Web Search** using [Tavily API](https://docs.tavily.com/)
- 🧠 **LangGraph**-based workflow for context retrieval and answer generation
- ✍️ **Edited NCERT-style answers** using Meta’s open-source **LLaMA 3.2 1B Instruct**
- 🗂️ **Memory-enabled chatbot**
- 💬 **Gradio UI** to interact with the model

---

## 🧠 LLM Used

### 🔹 Model: [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

- Lightweight, fast, and open-access instruction-tuned model by Meta
- Used for both generation and editing in this project
- Deployed with `transformers.pipeline()` for ease of integration

---

## 🛠️ Tools & Technologies

| Tool          | Purpose                                               |
|---------------|--------------------------------------------------------|
| `LangChain`   | Prompting, embeddings, vector search wrapper          |
| `LangGraph`   | Modular, state-based RAG pipeline (rag → web → LLM)   |
| `FAISS`       | Local vector database for PDF context retrieval       |
| `Tavily API`  | Real-time web search augmentation                     |
| `transformers`| Load LLaMA 3.2 1B Instruct from Hugging Face          |
| `Gradio`      | Build an interactive chatbot interface                |
| `PyPDFLoader` | Load and split NCERT PDF content                      |

---

## 🚀 How to Use (Locally)

1. Clone the repo and open the `.ipynb` notebook in Colab or Jupyter Lab.

2. Install required packages:
```bash
pip install -r requirements.txt
