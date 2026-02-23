import sys
from os import path
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# 1. Add your app to the sys.path so Alembic can find your models
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from app.core.config import settings
from app.core.database import Base

from app.models.users import User 
from app.models.emotions import Emotion

config = context.config

# 2. Setup logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 3. Set the target metadata for autogenerate
target_metadata = Base.metadata

def run_migrations_online():
    """
    Run migrations in 'online' mode.
    Uses the DATABASE_URL defined in our settings.
    """
    # Use the dynamic URL from our config
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = settings.DATABASE_URL

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata,
            compare_type=True 
        )

        with context.begin_transaction():
            context.run_migrations()

run_migrations_online()