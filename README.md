# AI-Powered Conversational Data Analytics Engine

## 🚀 Download & Run the Application

[![Download App](https://shields.io)](../../releases/latest)

### 💻 Quick Start Instructions (Windows)
1. Click the **Download** badge above to grab the `conversational_project.zip` file.
2. Right-click the downloaded ZIP file and select **"Extract All..."**.
3. Open the extracted folder and double-click the **`run.bat`** file (the file with the gear icon).
4. The app will automatically open right up in your web browser!

A chat app that answers plain-English questions about PDF and CSV files.
It runs fully on your own computer with Ollama, so no API keys are needed
and your files never leave your machine.

## Features
- Upload a PDF or CSV and ask questions in a chat window
- PDFs: text is extracted and the relevant parts are sent to a local LLM
- CSVs: Pandas calculates the answer, and the LLM explains the result
- Chat history and a clear button in the sidebar

## Tech Stack
Python, Streamlit, LangChain, Ollama, Pandas, PyPDF

## How to Run
1. Install [Ollama](https://ollama.com) and download the models:

        ollama pull llama3.1:8b
        ollama pull nomic-embed-text

2. Install the packages:

        pip install -r requirements.txt

3. Start the app:

        streamlit run app.py

To use a smaller model on a low-memory laptop, change `LLM_MODEL` at the top of `app.py` (for example to `llama3.2:3b`).


