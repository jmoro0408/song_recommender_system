import logging
import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
Base = declarative_base()


class PGManager:
    def __init__(self):
        self.db_name = os.getenv("POSTGRES_DB")
        self.user = os.getenv("POSTGRES_USER")
        self.password = os.getenv("POSTGRES_PASSWORD")
        self.hostname = os.getenv("POSTGRES_HOST")
        self.port = os.getenv("POSTGRES_PORT")
        self.engine = create_engine(
            f"postgresql+psycopg2://{self.user}:{self.password}@{self.hostname}:{self.port}/{self.db_name}"
        )
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.scoped_session = scoped_session(self.SessionFactory)
        Base.metadata.bind = self.engine

    @contextmanager
    def get_session(self):
        """Provide a transactional scope for ORM operations."""
        session = self.scoped_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error: {e}")
            raise
        finally:
            session.close()

    def create_all_tables(self, base):
        """Create all tables from an external Base."""
        base.metadata.create_all(self.engine)
        logger.info("All tables created successfully.")

    def drop_all_tables(self, base):
        """Drop all tables from an external Base.."""
        base.metadata.drop_all(self.engine)
        logger.info("All tables dropped successfully.")
