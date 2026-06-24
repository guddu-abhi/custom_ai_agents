from domain.models.search import ProductFilters
from retrieval.db.search_repo import build_filter_sql


def test_none_filters():
    assert build_filter_sql(None) == ("", {})


def test_empty_filters():
    clause, params = build_filter_sql(ProductFilters())
    assert clause == ""
    assert params == {}


def test_price_max_is_null_safe():
    # ~57% of products have NULL price; the clause must keep them rather than
    # let `NULL <= :price_max` (which is NULL, not true) drop them.
    clause, params = build_filter_sql(ProductFilters(price_max=50))
    assert clause == "AND (p.price IS NULL OR p.price <= :price_max)"
    assert params == {"price_max": 50.0}


def test_min_rating():
    clause, params = build_filter_sql(ProductFilters(min_rating=4))
    assert clause == "AND p.average_rating >= :min_rating"
    assert params == {"min_rating": 4.0}


def test_min_reviews():
    clause, params = build_filter_sql(ProductFilters(min_reviews=100))
    assert clause == "AND p.rating_number >= :min_reviews"
    assert params == {"min_reviews": 100}


def test_brand():
    clause, params = build_filter_sql(ProductFilters(brand="Sony"))
    assert clause == "AND p.store ILIKE :brand"
    assert params == {"brand": "%Sony%"}


def test_no_category_field():
    # category was removed as a hard filter (folded into the semantic query).
    assert "category" not in ProductFilters.model_fields


def test_combined_filters_emit_all():
    clause, params = build_filter_sql(
        ProductFilters(brand="Sony", price_max=50, min_rating=4, min_reviews=100)
    )
    assert clause.count("AND ") == 4
    assert set(params) == {"price_max", "min_rating", "min_reviews", "brand"}
