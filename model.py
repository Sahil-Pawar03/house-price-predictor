from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATASET = Path(__file__).parent / "houses.csv"
FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years", "distance_km", "garage"]
TARGET = "price"


def load_data():
    """Load housing dataset, generating it if it doesn't already exist."""
    if not DATASET.exists():
        from generate_data import generate
        generate(DATASET)
    return pd.read_csv(DATASET).dropna(subset=FEATURES + [TARGET])


def train_model(df):
    """Train linear regression model and compute evaluation metrics."""
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    y_test_arr = y_test.values

    # Avoid zero division in MAPE
    non_zero = y_test_arr > 0
    mape = float(np.mean(np.abs((y_test_arr[non_zero] - preds[non_zero]) / y_test_arr[non_zero])) * 100)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
        "MAPE": mape,
    }
    test = pd.DataFrame({"Actual price": y_test.values, "Predicted price": preds})
    return model, metrics, test


def predict_price(model, **inputs):
    """Predict house price given feature values."""
    row = pd.DataFrame([inputs])[FEATURES]
    return max(float(model.predict(row)[0]), 0.0)


def calculate_mortgage(
    price,
    down_payment_pct=0.20,
    interest_rate=0.065,
    loan_years=30,
    property_tax_rate=0.012,
    insurance_annual=1200,
):
    """
    Calculate monthly mortgage payment with principal, interest, taxes, and insurance (PITI).
    """
    price = max(float(price), 0.0)
    down_payment = price * down_payment_pct
    loan_amount = max(price - down_payment, 0.0)

    # Monthly interest and months
    r = interest_rate / 12.0
    n = loan_years * 12

    if loan_amount <= 0 or r <= 0:
        monthly_pi = 0.0
    else:
        # Standard amortization formula
        monthly_pi = loan_amount * (r * ((1 + r) ** n)) / (((1 + r) ** n) - 1)

    monthly_tax = (price * property_tax_rate) / 12.0
    monthly_insurance = insurance_annual / 12.0
    total_monthly = monthly_pi + monthly_tax + monthly_insurance

    return {
        "price": price,
        "down_payment": down_payment,
        "loan_amount": loan_amount,
        "monthly_principal_interest": monthly_pi,
        "monthly_tax": monthly_tax,
        "monthly_insurance": monthly_insurance,
        "total_monthly": total_monthly,
    }


def get_property_persona(area_sqft, bedrooms, bathrooms, price, distance_km):
    """
    Determine a descriptive persona and lifestyle badge based on home features.
    """
    if price >= 550_000 or area_sqft >= 3200 or (bedrooms >= 5 and bathrooms >= 3):
        return "🏰 Executive Luxury Estate"
    elif distance_km <= 5.0 and area_sqft <= 1400:
        return "🌆 Urban Metropolitan Pad"
    elif bedrooms >= 3 and area_sqft >= 1700:
        return "🏡 Spacious Family Residence"
    elif area_sqft < 1200 and bedrooms <= 2:
        return "🏠 Cozy Starter Cottage"
    else:
        return "🏘️ Modern Suburban Home"


if __name__ == "__main__":
    df = load_data()
    _, m, _ = train_model(df)
    print(f"Records: {len(df)}")
    print(f"MAE:  ${m['MAE']:,.0f}\nRMSE: ${m['RMSE']:,.0f}\nR2:   {m['R2']:.3f}\nMAPE: {m['MAPE']:.1f}%")
