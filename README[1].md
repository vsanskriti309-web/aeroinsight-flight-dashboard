# AeroInsight — Flight Fare Prediction

## Run in VS Code (Windows)
1. Open this folder in VS Code.
2. Choose **Terminal → New Terminal**.
3. Install packages: `python -m pip install -r requirements.txt`
4. Start the website: `python -m streamlit run app.py`
5. Open the Local URL shown in the terminal, usually http://localhost:8501.

Upload a CSV in the sidebar. It needs a numeric `Price` or `Fare` column. Helpful extra columns include `Airline`, `Source`, `Destination`, `Duration_Minutes`, and `Total_Stops`. The included sample data is synthetic practice data, not real ticket prices.
