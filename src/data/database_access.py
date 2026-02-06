from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, update as sql_update, delete as sql_delete
from typing import Type, Any, Dict, List, Optional, AsyncIterator
from contextlib import asynccontextmanager


class DatabaseAccess:
    
    def __init__(self, database_url: str, echo: bool = False):
        self.engine = create_async_engine(
            database_url,
            echo=echo,
            future=True,
            pool_pre_ping=True,
        )

        self.SessionLocal = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    # -------- Session --------

    @asynccontextmanager
    async def get_session(self) -> AsyncIterator[AsyncSession]:
        async with self.SessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # -------- CREATE --------

    async def insert(self, obj: Any) -> Any:
        async with self.get_session() as session:
            session.add(obj)
            await session.flush()
            await session.refresh(obj)
            return obj

    async def insert_many(self, objects: List[Any]) -> List[Any]:
        async with self.get_session() as session:
            session.add_all(objects)
            await session.flush()
            return objects

    # -------- READ --------

    async def get_by_id(self, model: Type, pk: Any) -> Optional[Any]:
        async with self.get_session() as session:
            return await session.get(model, pk)

    async def list(
        self,
        model: Type,
        filters = None,
        limit = None,
        offset = None,
    ):
        async with self.get_session() as session:
            stmt = select(model)

            if filters:
                stmt = stmt.filter_by(**filters)
            if limit:
                stmt = stmt.limit(limit)
            if offset:
                stmt = stmt.offset(offset)

            result = await session.execute(stmt)
            return result.scalars().all()

    async def first(self, model: Type, filters = None) -> Optional[Any]:
        async with self.get_session() as session:
            stmt = select(model)
            if filters:
                stmt = stmt.filter_by(**filters)

            result = await session.execute(stmt)
            return result.scalars().first()

    # -------- UPDATE --------

    async def update_by_id(
        self,
        model: Type,
        pk: Any,
        values: Dict,
    ):
        async with self.get_session() as session:
            pk_col = model.__mapper__.primary_key[0]

            stmt = (
                sql_update(model)
                .where(pk_col == pk)
                .values(**values)
            )

            result = await session.execute(stmt)
            
            return result

    # -------- DELETE --------

    async def delete_by_id(self, model: Type, pk: Any):
        async with self.get_session() as session:
            pk_col = model.__mapper__.primary_key[0]

            stmt = (
                sql_delete(model)
                .where(pk_col == pk)
            )

            result = await session.execute(stmt)
            return result
