import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Connection, Engine, create_engine

from loader.config import settings

_engines: dict[str, Engine] = {}


def _url_for_env(env: str) -> str:
    if env == "local":
        return settings.db_url
    env_var = f"LOADER_DB_URL_{env.upper()}"
    url = os.environ.get(env_var)
    if not url:
        raise ValueError(f"Environment variable {env_var!r} is not set for env={env!r}")
    return url


def get_engine(env: str) -> Engine:
    if env not in _engines:
        _engines[env] = create_engine(_url_for_env(env), echo=False, pool_pre_ping=True)
    return _engines[env]


@contextmanager
def get_connection(env: str) -> Iterator[Connection]:
    with get_engine(env).connect() as conn:
        yield conn
