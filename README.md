# HIRA Gemini Chatbot

Streamlit-based Tata Motors HIRA assistant that:
- Extracts HIRA fields using Gemini API for every user query
- Prioritizes BIW_MAINT.xlsx reference data when available
- Uses deterministic rule-based risk calculations (RPN)
- Generates a BIW-format Excel report

## Models used

- Default Gemini model: `gemini-2.0-flash`
- You can override with `GEMINI_MODEL` in `.env`

## Create a Gemini API key

1) Go to Google AI Studio: https://aistudio.google.com/app/apikey
2) Click "Create API key" and copy it.
3) In `.env` file in the project root and add:

```env
GEMINI_API_KEY=your_real_key_here
```

Optional:

```env
GEMINI_MODEL=gemini-2.0-flash
GEMINI_API_BASE=https://generativelanguage.googleapis.com/v1beta
```

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

## 3) Run tests

```powershell
python -m unittest tests/test_bot_engine.py
```

## 4) Run app

```powershell
python -m streamlit run src/chat_ui.py --server.port 8505
```

Open:
- http://localhost:8505

## Notes

- `.env` is loaded automatically by the app.
- `.env` is ignored by git.
- If API key is missing, the UI shows a warning.
