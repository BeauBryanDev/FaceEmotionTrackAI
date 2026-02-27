import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, Text, text as sql_text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.types import TypeDecorator
from sqlalchemy.pool import StaticPool
from contextlib import asynccontextmanager

# Import all models FIRST to ensure Base.metadata is populated
from app.core.database import Base
import app.models.users
import app.models.emotions
import app.models.face_session

# Now import the app
from app.main import app
from app.core.session import get_db
from app.models.users import User
from app.core.security import get_password_hash

# Replace the app's lifespan with a no-op version to prevent DB/ML initialization
@asynccontextmanager
async def _empty_lifespan(_app):
    """Empty lifespan for testing - skips DB init and ML model loading."""
    yield

# CRITICAL: Override the lifespan context manager BEFORE any TestClient is created
app.router.lifespan_context = _empty_lifespan


class VectorAsText(TypeDecorator):
    """
    SQLite-compatible replacement for pgvector's Vector type.
    Stores the embedding as a JSON string in SQLite.
    In production PostgreSQL, the real Vector(512) type is used.
    """
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):

        if value is None:

            return None

        import json

        return json.dumps(value) if isinstance(value, list) else str(value)


    def process_result_value(self, value, dialect):

        if value is None:

            return None

        import json

        try:

            return json.loads(value)

        except (ValueError, TypeError):

            return value


def _patch_vector_columns():
    """
    Patches all PostgreSQL-specific column types to SQLite-compatible types.
    Models are already imported at the top of the file to ensure their
    mappers are registered before iterating Base.registry.mappers.
    """
    from pgvector.sqlalchemy import Vector
    from sqlalchemy.dialects.postgresql import JSONB

    for mapper in Base.registry.mappers:
        for column in mapper.mapped_table.columns:
            if isinstance(column.type, Vector):
                column.type = VectorAsText()
            elif isinstance(column.type, JSONB):
                column.type = Text()

# Fixtures

# Use a SHARED in-memory database so all connections see the same data
# The key is "cache=shared" which makes the database accessible across connections
SQLITE_URL = "sqlite:///:memory:?cache=shared"


@pytest.fixture(scope="function")
def db_session():
    """
    Creates a fresh SQLite in-memory database for each test function.
    Uses a shared in-memory database so all connections see the same tables.
    """
    # Patch BEFORE creating engine and tables
    _patch_vector_columns()

    engine = create_engine(
        SQLITE_URL,
        connect_args={"check_same_thread": False, "uri": True},
        poolclass=StaticPool,  # Use a single connection pool
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # Create all tables with patched column types
    Base.metadata.create_all(bind=engine)

    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=engine
    )
    session = TestingSessionLocal()
    try:

        yield session

    finally:

        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture(scope="function")
def client(db_session):
    """
    FastAPI TestClient with the get_db dependency overridden
    to use the SQLite in-memory session from db_session fixture.
    """
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def registered_user_payload() -> dict:
    """Valid registration payload for a test user."""
    return {
        "full_name": "Test User",
        "email"    : "testuser@example.com",
        "password" : "SecurePass123",
        "age"      : 25
    }


@pytest.fixture
def registered_user(client, registered_user_payload) -> dict:
    """
    Registers a user via the API and returns the response JSON.
    Used as a base fixture for tests that require an existing user.
    """
    response = client.post(
        "/api/v1/auth/register",
        json=registered_user_payload
    )
    assert response.status_code == 201, (
        f"Setup failed: could not register test user. "
        f"Response: {response.json()}"
    )
    return response.json()


@pytest.fixture
def auth_token(client, registered_user_payload, registered_user) -> str:
    """
    Logs in the registered test user and returns the JWT access token.
    Used as a base fixture for authenticated endpoint tests.
    """
    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": registered_user_payload["email"],
            "password": registered_user_payload["password"],
        }
    )
    assert response.status_code == 200, (
        f"Setup failed: could not login test user. "
        f"Response: {response.json()}"
    )
    return response.json()["access_token"]


@pytest.fixture
def auth_headers(auth_token) -> dict:
    """Returns Authorization header dict for authenticated requests."""
    return {"Authorization": f"Bearer {auth_token}"}


# TEST REGISTER
# POST /api/v1/auth/register

class TestRegister:
    """Integration tests for POST /api/v1/auth/register."""

    def test_register_success_returns_201(
        self, client, registered_user_payload
    ):
        """
        Valid registration payload must return HTTP 201 Created.
        """
        response = client.post(
            "/api/v1/auth/register",
            json=registered_user_payload
        )
        assert response.status_code == 201

    def test_register_returns_user_fields(
        self, client, registered_user_payload
    ):
        """
        Response body must contain id, full_name, email.
        Must NOT contain hashed_password (security requirement).
        """
        response = client.post(
            "/api/v1/auth/register",
            json=registered_user_payload
        )
        data = response.json()
        assert "id"        in data
        assert "full_name" in data
        assert "email"     in data
        assert "hashed_password" not in data

    def test_register_duplicate_email_returns_400(
        self, client, registered_user_payload, registered_user
    ):
        """
        Registering with an email that already exists must return 400.
        The registered_user fixture already created the first account.
        """
        response = client.post(
            "/api/v1/auth/register",
            json=registered_user_payload
        )
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"].lower()

    def test_register_missing_required_field_returns_422(self, client):
        """
        Payload missing required fields must return 422 Unprocessable Entity.
        FastAPI validates the Pydantic schema before hitting the endpoint.
        """
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "incomplete@example.com"}
        )
        assert response.status_code == 422

    def test_register_invalid_email_format_returns_422(self, client):
        """
        Invalid email format must be rejected by Pydantic validation.
        """
        response = client.post(
            "/api/v1/auth/register",
            json={
                "full_name": "Test User",
                "email"    : "not-an-email",
                "password" : "SecurePass123",
                "age"      : 25
            }
        )
        assert response.status_code == 422


# TEST LOGIN
# POST /api/v1/auth/login

class TestLogin:
    """Integration tests for POST /api/v1/auth/login."""

    def test_login_success_returns_200(
        self, client, registered_user, registered_user_payload
    ):
        """
        Valid credentials must return HTTP 200 with access_token.
        """
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": registered_user_payload["email"],
                "password": registered_user_payload["password"],
            }
        )
        assert response.status_code == 200

    def test_login_returns_token_and_type(
        self, client, registered_user, registered_user_payload
    ):
        """
        Response must contain access_token and token_type='bearer'.
        Contract required by stream.py WebSocket auth via query param.
        """
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": registered_user_payload["email"],
                "password": registered_user_payload["password"],
            }
        )
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert len(data["access_token"]) > 0

    def test_login_wrong_password_returns_401(
        self, client, registered_user, registered_user_payload
    ):
        """
        Wrong password must return 401 Unauthorized.
        """
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": registered_user_payload["email"],
                "password": "WrongPassword999",
            }
        )
        assert response.status_code == 401

    def test_login_nonexistent_email_returns_401(self, client):
        """
        Login attempt with an email not in the database must return 401.
        Must not reveal whether the email exists (security best practice).
        """
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "ghost@example.com",
                "password": "SomePassword123",
            }
        )
        assert response.status_code == 401

    def test_login_inactive_user_returns_400(
        self, client, db_session, registered_user_payload
    ):
        """
        A deactivated user account must not be able to login.
        Simulates an admin-disabled account.
        """
        # Register the user first
        client.post("/api/v1/auth/register", json=registered_user_payload)

        # Deactivate the user directly in the test DB
        user = db_session.query(User).filter(
            User.email == registered_user_payload["email"]
        ).first()
        user.is_active = False
        db_session.commit()

        # Attempt login
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": registered_user_payload["email"],
                "password": registered_user_payload["password"],
            }
        )
        assert response.status_code == 400


# TEST PROTECTED ROUTE - GET /api/v1/users/me
# Verifies the JWT token produced by login is accepted by protected endpoints.

class TestProtectedRoute:
    """Verifies JWT token flow from login to protected endpoint access."""

    def test_authenticated_request_returns_200(
        self, client, registered_user, auth_headers
    ):
        """
        A valid JWT token in the Authorization header must grant
        access to the protected GET /api/v1/users/me endpoint.
        """
        response = client.get("/api/v1/users/me", headers=auth_headers)
        assert response.status_code == 200

    def test_request_without_token_returns_401(self, client, registered_user):
        """
        A request to a protected endpoint without Authorization header
        must return 401 Unauthorized.
        """
        response = client.get("/api/v1/users/me")
        assert response.status_code == 401

    def test_request_with_invalid_token_returns_401(self, client):
        """
        A malformed or expired JWT must be rejected with 401.
        """
        response = client.get(
            "/api/v1/users/me",
            headers={"Authorization": "Bearer this.is.not.a.valid.jwt"}
        )
        assert response.status_code == 401
