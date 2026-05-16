# env.py

import asyncio
from logging.config import fileConfig
from alembic import context
from sqlalchemy import MetaData, text
from sqlalchemy.orm import declarative_base
from utils.db_utils import WebAppDBFactory
from orm import CatalogBase

# this is the Alembic Config object
config = context.config
connectable = context.config.attributes.get("connection", None)

# Interpret config file for Python logging
fileConfig(config.config_file_name)  # type: ignore[arg-type]

# set target metadata for 'autogenerate' support.
# Can also be a list of metadata if there are multiple bases
# This is set to None because we don't want to autogenerate
metadata = MetaData(schema="catalog")
CATALOG_BASE = declarative_base(metadata=metadata, cls=CatalogBase)  # type: ignore[assignment]
target_metadata = CATALOG_BASE.metadata

if connectable is None:
    connectable = WebAppDBFactory.get_db_engine(env="local")  # type: ignore[call-arg]


def include_name(name, type_, parent_names):
    if type_ == "schema":
        return name == target_metadata.schema
    return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    context.configure(
        url=connectable.url,
        target_metadata=target_metadata,
        version_table_schema=target_metadata.schema,
        literal_binds=True,
        include_schemas=True,
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:  # noqa: ANN001
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table_schema=target_metadata.schema,
        include_schemas=True,
        include_name=include_name,
    )

    with context.begin_transaction():

        connection.execute(
            text(
                f"""
                CREATE SCHEMA IF NOT EXISTS {target_metadata.schema};
                """
            )  # noqa: S608
        )

        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
