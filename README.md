# HIRA Gemini Chatbot

A Streamlit-based Tata Motors HIRA assistant using:
- Gemini API for field extraction on every user query
- Deterministic rule-based risk calculations (RPN)

## 1) Create and activate virtual environment

```powershell
cd "d:\HIRA ai"
C:/Users/Aniruddh/AppData/Local/Programs/Python/Python311/python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2) Install dependencies

```powershell
pip install -r requirements.txt
```

## 3) Add Gemini API key

Open `.env` and set:

```env
GEMINI_API_KEY=your_real_key_here
```

Optional:

```env
GEMINI_MODEL=gemini-2.0-flash
GEMINI_API_BASE=https://generativelanguage.googleapis.com/v1beta
```

## 4) Run tests

```powershell
python -m unittest tests/test_bot_engine.py
```

## 5) Run app

```powershell
python -m streamlit run src/chat_ui.py --server.port 8505
```

Open:
- http://localhost:8505

## Notes

- `.env` is loaded automatically by the app.
- `.env` is ignored by git.
- If API key is missing, the UI shows a warning.
