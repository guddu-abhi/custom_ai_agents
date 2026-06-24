import asyncio

from pydantic import BaseModel
from sqlalchemy import text

from agents import function_tool
from otto_lib.db.session import db_session


class ProductDescriptionResponse(BaseModel):
    description: str
    specs: str


@function_tool
async def get_product_availability(category: str) -> list[str] | None:
    """Get product availability based on category."""
    async with db_session() as session:
        result = await session.execute(
            text("SELECT name FROM products WHERE lower(category) = :category"),
            {"category": category.lower()},
        )
        products = result.fetchall()
        if products:
            return [product[0] for product in products]
        else:
            return None


@function_tool
async def get_product_information_by_name(name: str) -> tuple[str, str] | list[ProductDescriptionResponse] | None:
    """Get product information based on name."""
    async with db_session() as session:
        result = await session.execute(
            text("SELECT description, specs FROM products WHERE lower(name) = :name"),
            {"name": name.lower()},
        )
        product = result.fetchone()
        if product:
            return product
        else:
            return None



if __name__ == "__main__":
    async def main():
        products = await get_product_availability("smartphone")
        print(products)

    asyncio.run(main())
