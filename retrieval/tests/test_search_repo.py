"""Integration test skeleton for SearchRepository.

Marked `skip` until the testcontainers Postgres+pgvector fixture lands
(tracked in arch_plan/refactor_plan.md). Mirrors the pattern reserved for
test_product_repo.py.
"""
import pytest

pytest.skip("requires testcontainers postgres+pgvector fixture", allow_module_level=True)
