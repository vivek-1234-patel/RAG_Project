# DocuChat Streamlit RAG App

## Install
Copy `app.py` and `requirements.txt` into your existing RAG project folder:
`D:\DATA ANALYSIS\Generative_AI\RAG`

Activate your existing virtual environment, then run:
```powershell
python -m pip install -r requirements.txt
```

Ensure the project `.env` contains either `GEMINI_API_KEY=your_key` or `GOOGLE_API_KEY=your_key`.

## Run
From the RAG project folder:
```powershell
python -m streamlit run app.py
```
Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Features
- Upload PDF or TXT documents
- Split, embed using `models/gemini-embedding-001`, and index the uploaded file
- Chat UI with MMR retrieval and expandable retrieved source snippets
- Uses an in-memory Chroma collection to avoid mixing uploaded documents or embedding spaces; re-index after app restart

If `gemini-3.8-flash` is unavailable for your API key, change that model ID to one currently supported by your Gemini API.
Scanned/image-only PDFs may require OCR.
