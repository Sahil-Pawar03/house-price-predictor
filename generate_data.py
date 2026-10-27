"""Creates a synthetic housing dataset (houses.csv) so the project runs out of the box."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate(path="houses.csv", n=1000, seed=42):
    rng = np.random.default_rng(seed)

    area = rng.integers(500, 4000, n)
    bedrooms = np.clip((area / 900 + rng.normal(0, 0.8, n)).round(), 1, 6).astype(int)
    bathrooms = np.clip((bedrooms * 0.6 + rng.normal(0, 0.6, n)).round(), 1, 5).astype(int)
    age = rng.integers(0, 50, n)
    distance = rng.uniform(1, 30, n).round(1)
    garage = rng.integers(0, 3, n)

    price = (
        40_000
        + 140 * area
        + 8_000 * bedrooms
        + 14_000 * bathrooms
        - 1_100 * age
        - 3_200 * distance
        + 11_000 * garage
        + rng.normal(0, 22_000, n)   # real-world noise
    ).clip(30_000)

    df = pd.DataFrame({
        "area_sqft": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age,
        "distance_km": distance,
        "garage": garage,
        "price": price.round(0).astype(int),
    })
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    out = Path(__file__).parent / "houses.csv"
    generate(out)
    print(f"Saved {out}")
