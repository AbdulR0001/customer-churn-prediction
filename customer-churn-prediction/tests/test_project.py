from src.data import FEATURES, TARGET, generate_customer_data

def test_data_generation():
    df = generate_customer_data(40, 7)
    assert len(df) == 40
    assert set(FEATURES + [TARGET]).issubset(df.columns)
    assert set(df[TARGET].unique()).issubset({0, 1})
