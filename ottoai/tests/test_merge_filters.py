from domain.models.search import ProductFilters
from ottoai.filters import merge_filters


def test_overlay_adds_new_filter_keeps_existing():
    current = ProductFilters(price_max=50, min_rating=4.5, min_reviews=100)
    merged = merge_filters(current, ProductFilters(brand="Sony"), reset=False)
    assert merged.model_dump(exclude_none=True) == {
        "brand": "Sony",
        "price_max": 50.0,
        "min_rating": 4.5,
        "min_reviews": 100,
    }


def test_overlay_changes_one_field_carries_the_rest():
    # "cheaper ones": lower price_max, keep rating/reviews.
    current = ProductFilters(price_max=50, min_rating=4.5, min_reviews=100)
    merged = merge_filters(current, ProductFilters(price_max=25), reset=False)
    assert merged.price_max == 25.0
    assert merged.min_rating == 4.5
    assert merged.min_reviews == 100


def test_null_delta_field_keeps_carried_value():
    current = ProductFilters(brand="Sony", price_max=50)
    merged = merge_filters(current, ProductFilters(), reset=False)
    assert merged.model_dump(exclude_none=True) == {"brand": "Sony", "price_max": 50.0}


def test_reset_drops_everything_and_uses_delta():
    current = ProductFilters(brand="Sony", price_max=50, min_rating=4.5)
    merged = merge_filters(current, ProductFilters(brand="Anker"), reset=True)
    assert merged.model_dump(exclude_none=True) == {"brand": "Anker"}


def test_reset_with_empty_delta_clears_all():
    current = ProductFilters(brand="Sony", price_max=50)
    merged = merge_filters(current, ProductFilters(), reset=True)
    assert merged.model_dump(exclude_none=True) == {}
