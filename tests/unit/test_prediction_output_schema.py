from market_slice_ml.domain.models import PredictionRecord


def test_prediction_schema_has_exactly_six_research_fields():
    prediction = PredictionRecord(
        symbol="NQ=F",
        horizon="h1",
        direction="neutral",
        expected_return=0.0,
        expected_volatility=0.2,
        confidence_score=50.0,
    )
    assert set(prediction.model_dump()) == {
        "symbol",
        "horizon",
        "direction",
        "expected_return",
        "expected_volatility",
        "confidence_score",
    }
