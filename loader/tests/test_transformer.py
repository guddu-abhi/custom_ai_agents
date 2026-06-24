from loader.core.transformer import PipelineRow, transform


def _record(**overrides):
    base = {
        "main_category": "Electronics",
        "title": "Wireless Headphones",
        "store": "AudioCo",
        "description": ["Great sound", "Long battery"],
        "features": ["Noise cancelling", "Bluetooth 5.0"],
        "categories": ["Electronics", "Audio"],
        "average_rating": 4.5,
        "rating_number": 1200,
        "price": "49.99",
        "parent_asin": "B001TEST",
        "details": {"Date First Available": "January 15, 2024"},
    }
    base.update(overrides)
    return base


def test_transform_returns_pipeline_row():
    result = transform(_record())
    assert isinstance(result, PipelineRow)


def test_embedding_content_joins_fields():
    result = transform(_record())
    assert result is not None
    content = result.embedding_content
    assert "Electronics" in content
    assert "Wireless Headphones" in content
    assert "AudioCo" in content
    assert "Great sound" in content


def test_description_list_joined_with_newline():
    result = transform(_record())
    assert result is not None
    assert result.db_row["description"] == "Great sound\nLong battery"


def test_filters_pre_min_year():
    result = transform(_record(details={"Date First Available": "March 5, 2020"}), min_year=2023)
    assert result is None


def test_keeps_record_at_min_year():
    result = transform(_record(details={"Date First Available": "January 1, 2023"}), min_year=2023)
    assert result is not None


def test_keeps_record_with_no_date():
    result = transform(_record(details={}), min_year=2023)
    assert result is not None


def test_features_defaults_to_empty_list_when_missing():
    result = transform(_record(features=None))
    assert result is not None
    assert result.db_row["features"] == []


def test_categories_defaults_to_empty_list_when_missing():
    result = transform(_record(categories=None))
    assert result is not None
    assert result.db_row["categories"] == []


def test_price_parsed_as_decimal():
    from decimal import Decimal
    result = transform(_record(price="1,299.00"))
    assert result is not None
    assert result.db_row["price"] == Decimal("1299.00")


def test_price_none_when_invalid():
    result = transform(_record(price="N/A"))
    assert result is not None
    assert result.db_row["price"] is None


def test_embedding_content_sanitized():
    result = transform(_record(title="Headphones\x00Clean"))
    assert result is not None
    assert "\x00" not in result.embedding_content
