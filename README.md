# Car Resale Price Prediction — Lasso Regression

Day 5 of the daily ML project series: FastAPI backend + Streamlit frontend.

## Algorithm
**Lasso Regression** (L1-regularized linear regression) with `StandardScaler`. Unlike
Ridge, Lasso can shrink coefficients exactly to zero — meaning it performs automatic
feature selection.

## Dataset — built specifically to demonstrate Lasso
`data/car_resale_data.csv` is a **synthetic dataset generated for this project**
(`data/generate_dataset.py`-style logic, see `backend/train.py` comments), deliberately
designed with:
- **5 real signal features**: `Age_Years`, `Mileage_km`, `Engine_CC`, `Owner_Count`,
  `Accident_History` — these actually drive `Resale_Price_USD` in the generating formula.
- **3 irrelevant/noise features**: `Color_Popularity_Score`, `Local_Fuel_Price_Index`,
  `Service_Center_Visits` — random values with no real effect on price, mixed in on purpose.

This is the whole point of the project: Lasso should learn to zero out the 3 irrelevant
features while keeping the 5 real ones. The frontend has a live "What Lasso Actually Uses"
panel that visualizes exactly this using the `/feature-importance` endpoint.

## Model Performance (on the 20% held-out test split)
- MAE: ~655
- R² Score: ~0.99
- Learned coefficients (scaled): all 3 noise features shrink to exactly 0.0 / -0.0

## Project Structure
```
car-resale-lasso/
├── .streamlit/
│   └── config.toml           # dark "midnight showroom" theme
├── data/
│   └── car_resale_data.csv
├── backend/
│   ├── train.py                # dataset EDA + training, saves model_lasso.pkl + feature_importance.json
│   ├── main.py                   # FastAPI app — /predict and /feature-importance endpoints
│   └── schema.py                  # Pydantic request/response models
├── frontend/
│   └── app.py                      # Streamlit UI — dark automotive digital dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the model (from `backend/`):
   ```
   cd backend
   python train.py
   ```
   This saves both `model_lasso.pkl` and `feature_importance.json` (used by the
   `/feature-importance` endpoint).

3. Start the FastAPI backend (from `backend/`):
   ```
   uvicorn main:app --reload --port 8000
   ```

4. Start the Streamlit frontend (from `frontend/`, in a separate terminal):
   ```
   streamlit run app.py
   ```

## Deploying (same pattern as previous days)
- Backend → Render. Build command: `pip install -r requirements.txt && cd backend && python train.py`.
  Start command: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`.
- Frontend → Streamlit Community Cloud. Main file path: `frontend/app.py`.
  Set the `API_URL` secret to your Render backend's `/predict` URL — the frontend derives
  the feature-importance URL from it automatically.
