# 🏡 HomeVal AI — Smart Real Estate Valuation & Insights (v2.5 Pro)

An intelligent, explainable real-estate appraisal platform powered by Machine Learning (scikit-learn Linear Regression) and an ultra-modern Streamlit interface.

---

## ✨ Features & Capabilities

- 🤖 **Instant Machine Learning Valuation**: Predicts market price in real-time with continuous confidence boundaries ($\pm \text{MAE}$).
- 🏷️ **Dynamic Property Archetype**: Intelligently tags properties (e.g. *🏰 Executive Luxury Estate*, *🏡 Spacious Family Residence*, *🌆 Urban Metropolitan Pad*).
- 🔍 **Explainable AI (Value Drivers)**: Visual contribution breakdown showing exact monetary and percentage impacts of living area, bedrooms, bathrooms, garage, property age, and proximity to downtown.
- 💳 **Integrated PITI Mortgage Estimator**: Instant monthly affordability calculation including Principal & Interest, estimated Property Taxes (1.2%/yr), and Homeowner Insurance.
- ⚖️ **Scenario Comparison Studio**: Save up to 4 property configurations to compare specs, estimated valuations, price-per-square-foot, and monthly costs side-by-side.
- 💱 **Multi-Currency Support**: Instant toggle between USD (`$`), EUR (`€`), GBP (`£`), and INR (`₹`).
- 📈 **Model Diagnostics & Error Analytics**: Real-time evaluation metrics ($R^2$ score, MAE, RMSE, MAPE) with predicted-vs-actual scatter charts and marginal feature coefficient tables.
- 📊 **Interactive Market Data Explorer**: Multi-dimensional filtering by price, bedroom count, and living area with live dataset KPIs and CSV export.
- 🧪 **Full Automated Test Suite**: Comprehensive unit and domain-monotonicity tests with `pytest`.

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Sahil-Pawar03/house-price-predictor.git
cd house-price-predictor

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### 3. Run Automated Tests
```bash
pytest test_model.py -v
```

---

## 📂 Project Architecture

```
house-price-predictor-v2/
├── app.py               # Streamlit application with modern UI & reactive components
├── model.py             # ML pipeline, mortgage calculator & persona classifier
├── generate_data.py     # Generates reproducible synthetic housing data (houses.csv)
├── test_model.py        # Automated test suite (unit, regression, monotonicity)
├── requirements.txt     # Python dependencies
├── houses.csv           # Baseline synthetic housing dataset
└── .streamlit/
    └── config.toml      # Modern visual theme and browser settings
```

---

## 📊 Model & Features

The model learns linear weights across 6 key physical and geographical housing attributes:
- `area_sqft`: Finished indoor living area
- `bedrooms`: Number of bedrooms
- `bathrooms`: Number of full bathrooms
- `age_years`: Structure age in years
- `distance_km`: Proximity to the central business district (km)
- `garage`: Private vehicle parking spaces

---

## 📜 License
MIT License. Open for educational and practical machine learning exploration.
