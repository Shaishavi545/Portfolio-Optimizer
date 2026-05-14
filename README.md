# Portfolio Optimizer

A Python-based financial portfolio optimization tool leveraging Markowitz Mean-Variance Analysis and live AI news sentiment scoring.

## Live Demo
[View Live Application](#) *(Update with your actual Render URL after deployment)*

## Project Structure

```text
Portfolio-Optimizer/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Pinned Python dependencies
├── .gitignore              # Git ignore file
├── LICENSE                 # MIT License
├── .streamlit/
│   └── config.toml         # Custom Streamlit theme configuration
│
├── data/
│   ├── __init__.py
│   └── nifty50.py          # Nifty 50 constituents and sector mappings
│
└── news/
    ├── __init__.py
    ├── fetcher.py          # RSS feed aggregator
    ├── macro.py            # Macroeconomic event detection
    └── sentiment.py        # FinBERT sentiment analysis engine
```

## How to Run Locally

1. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to the local server address provided in the terminal (usually `http://localhost:8501`).
