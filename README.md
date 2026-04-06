# 🛡️ Navi-Shield — Predictive Supply Chain Resilience

## Project
**Predictive Supply Chain Resilience: Navigating Disruptions via Custom Risk Variables**

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your CSVs in the same folder as app.py
```
navi_shield/
├── app.py
├── requirements.txt
├── sheet_1_conflict_data.csv
├── sheet_2_piracy_data.csv
├── sheet_3_sanctions_data-3.csv
└── sheet_4_embargo_data-4.csv
```

### 3. Run locally
```bash
streamlit run app.py
```

### 4. Deploy to Streamlit Community Cloud (free)
1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect your repo → select `app.py` → Deploy

## GRV Formula
```
GRV = 0.35 × conflict_score + 0.25 × piracy_score + 0.25 × sanction_score + 0.15 × embargo_score
```

## Sheet 5 Output
The pipeline auto-generates Sheet 5 (GRV Master). Download it from the **Export Sheet 5** tab.

## Modes
| Mode | Description |
|---|---|
| 📊 GRV Dashboard | Select any quarter, see recommended route + radar chart |
| 📈 Time Series | GRV trends across all quarters per route |
| 🤖 ML Prediction | RandomForest recommender + GBR next-quarter forecast |
| 📥 Export Sheet 5 | Download the full GRV Master CSV |
