"""
Unit and integration tests for the House Price Prediction model,
data generator, mortgage calculations, and utility functions.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from generate_data import generate
from model import (
    FEATURES,
    TARGET,
    calculate_mortgage,
    get_property_persona,
    load_data,
    predict_price,
    train_model,
)


def test_generate_data(tmp_path):
    """Test synthetic data generator creates valid CSV and dataframe."""
    csv_file = tmp_path / "test_houses.csv"
    df = generate(path=csv_file, n=50, seed=123)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert csv_file.exists()

    for feat in FEATURES + [TARGET]:
        assert feat in df.columns
        assert df[feat].notna().all()

    # Domain constraints
    assert (df["area_sqft"] >= 500).all()
    assert (df["bedrooms"] >= 1).all()
    assert (df["bathrooms"] >= 1).all()
    assert (df["price"] > 0).all()


def test_load_data():
    """Test load_data retrieves valid dataset."""
    df = load_data()
    assert not df.empty
    assert all(col in df.columns for col in FEATURES + [TARGET])
    assert df[TARGET].min() > 0


def test_train_model():
    """Test model training produces coefficients, valid metrics, and predictions."""
    df = load_data()
    model, metrics, test = train_model(df)

    # Check model
    assert hasattr(model, "coef_")
    assert len(model.coef_) == len(FEATURES)
    assert hasattr(model, "intercept_")

    # Check metrics
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert "MAPE" in metrics
    assert metrics["MAE"] > 0
    assert metrics["RMSE"] >= metrics["MAE"]  # Mathematical property: RMSE >= MAE
    assert 0.0 <= metrics["R2"] <= 1.0
    assert metrics["MAPE"] > 0

    # Check test predictions
    assert isinstance(test, pd.DataFrame)
    assert "Actual price" in test.columns
    assert "Predicted price" in test.columns
    assert len(test) > 0


def test_predict_price():
    """Test single house prediction and monotonic domain sanity checks."""
    df = load_data()
    model, _, _ = train_model(df)

    base = dict(area_sqft=1800, bedrooms=3, bathrooms=2, age_years=10, distance_km=8.0, garage=1)
    base_price = predict_price(model, **base)
    assert base_price > 50000

    # Monotonicity checks:
    # 1. Increasing area should increase price
    bigger = dict(base, area_sqft=3000)
    assert predict_price(model, **bigger) > base_price

    # 2. Older house should decrease price
    older = dict(base, age_years=35)
    assert predict_price(model, **older) < base_price

    # 3. Further distance from city centre should decrease price
    further = dict(base, distance_km=25.0)
    assert predict_price(model, **further) < base_price

    # 4. Adding garage should increase price
    with_garage = dict(base, garage=2)
    assert predict_price(model, **with_garage) >= base_price


def test_calculate_mortgage():
    """Test mortgage calculator math and edge conditions."""
    # $400,000 house, 20% down ($80,000 down, $320,000 loan)
    res = calculate_mortgage(
        price=400_000,
        down_payment_pct=0.20,
        interest_rate=0.065,
        loan_years=30,
        property_tax_rate=0.012,
        insurance_annual=1200,
    )

    assert res["loan_amount"] == 320_000
    assert res["down_payment"] == 80_000
    assert res["monthly_principal_interest"] > 1800 and res["monthly_principal_interest"] < 2300
    assert res["monthly_tax"] == 400
    assert res["monthly_insurance"] == 100
    assert res["total_monthly"] == pytest.approx(
        res["monthly_principal_interest"] + res["monthly_tax"] + res["monthly_insurance"], rel=1e-3
    )

    # 100% down payment (no loan)
    cash_deal = calculate_mortgage(price=300_000, down_payment_pct=1.0)
    assert cash_deal["loan_amount"] == 0
    assert cash_deal["monthly_principal_interest"] == 0


def test_property_persona():
    """Test dynamic property persona classification."""
    luxury = get_property_persona(area_sqft=3800, bedrooms=5, bathrooms=4, price=650000, distance_km=3.0)
    assert "Luxury" in luxury or "Estate" in luxury

    starter = get_property_persona(area_sqft=850, bedrooms=2, bathrooms=1, price=180000, distance_km=15.0)
    assert "Starter" in starter or "Cozy" in starter

    urban = get_property_persona(area_sqft=1100, bedrooms=2, bathrooms=1, price=280000, distance_km=2.5)
    assert "Urban" in urban or "City" in urban or "Starter" in urban
