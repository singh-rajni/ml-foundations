from ml_foundations.data import FEATURE_COLUMNS, make_incident_priority_data, temporal_split


def test_synthetic_data_is_reproducible_and_temporal():
    first = make_incident_priority_data(n_samples=600, seed=4)
    second = make_incident_priority_data(n_samples=600, seed=4)
    assert first.equals(second)
    assert set(FEATURE_COLUMNS).issubset(first.columns)
    train, validation, test = temporal_split(first)
    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()
