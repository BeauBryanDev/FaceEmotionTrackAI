import pytest


# GET /api/v1/users/me

class TestGetCurrentUser:
    """Integration tests for GET /api/v1/users/me."""

    def test_get_me_returns_200(
        self, client, registered_user, auth_headers
    ):
        """Authenticated request must return 200."""
        response = client.get("/api/v1/users/me", headers=auth_headers)
        assert response.status_code == 200

    def test_get_me_returns_correct_user_data(
        self, client, registered_user, auth_headers, registered_user_payload
    ):
        """
        Response must contain the authenticated user's full_name and email.
        Must not expose hashed_password.
        """
        response = client.get("/api/v1/users/me", headers=auth_headers)
        data = response.json()
        assert data["email"]     == registered_user_payload["email"]
        assert data["full_name"] == registered_user_payload["full_name"]
        assert "hashed_password" not in data

    def test_get_me_requires_authentication(self, client):
        """Request without token must return 401."""
        response = client.get("/api/v1/users/me")
        assert response.status_code == 401


# PUT /api/v1/users/me

class TestUpdateCurrentUser:
    """Integration tests for PUT /api/v1/users/me."""

    def test_update_full_name_returns_200(
        self, client, registered_user, auth_headers
    ):
        """
        Updating full_name with a valid payload must return 200.
        """
        response = client.put(
            "/api/v1/users/me",
            json={"full_name": "Updated Name"},
            headers=auth_headers
        )
        assert response.status_code == 200

    def test_update_full_name_persists(
        self, client, registered_user, auth_headers
    ):
        """
        After a successful PUT, GET /me must return the updated full_name.
        Verifies the change was committed to the database.
        """
        client.put(
            "/api/v1/users/me",
            json={"full_name": "Persisted Name"},
            headers=auth_headers
        )
        response = client.get("/api/v1/users/me", headers=auth_headers)
        assert response.json()["full_name"] == "Persisted Name"

    def test_update_duplicate_email_returns_400(
        self, client, db_session, registered_user, auth_headers,
        registered_user_payload
    ):
        """
        Updating email to one already used by another account must return 400.
        """
        # Register a second user
        second_payload = {
            "full_name": "Second User",
            "email"    : "second@example.com",
            "password" : "SecurePass456",
            "age"      : 30
        }
        client.post("/api/v1/auth/register", json=second_payload)

        # Try to update first user's email to second user's email
        response = client.put(
            "/api/v1/users/me",
            json={"email": "second@example.com"},
            headers=auth_headers
        )
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()

    def test_update_requires_authentication(self, client):
        """Request without token must return 401."""
        response = client.put(
            "/api/v1/users/me",
            json={"full_name": "No Auth"}
        )
        assert response.status_code == 401

    def test_update_with_empty_body_returns_200(
        self, client, registered_user, auth_headers
    ):
        """
        PUT with an empty body must return 200 without modifying any fields.
        UserUpdate uses exclude_unset=True so no fields are changed.
        """
        response = client.put(
            "/api/v1/users/me",
            json={},
            headers=auth_headers
        )
        assert response.status_code == 200



# DELETE /api/v1/users/me/face_embedding


class TestDeleteFaceEmbedding:
    """Integration tests for DELETE /api/v1/users/me/face_embedding."""

    def test_delete_face_embedding_returns_200(
        self, client, registered_user, auth_headers
    ):
        """
        Authenticated request must return 200 with a success message.
        Works even if face_embedding is already None.
        """
        response = client.delete(
            "/api/v1/users/me/face_embedding",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert "removed" in response.json()["message"].lower()

    def test_delete_face_embedding_requires_authentication(self, client):
        """Request without token must return 401."""
        response = client.delete("/api/v1/users/me/face_embedding")
        assert response.status_code == 401



# DELETE /api/v1/users/me


class TestDeleteCurrentUser:
    """Integration tests for DELETE /api/v1/users/me."""

    def test_delete_account_returns_204(
        self, client, registered_user, auth_headers
    ):
        """
        Authenticated DELETE must return 204 No Content.
        """
        response = client.delete("/api/v1/users/me", headers=auth_headers)
        assert response.status_code == 204

    def test_deleted_user_cannot_login(
        self, client, registered_user, auth_headers, registered_user_payload
    ):
        """
        After account deletion, login with the same credentials must fail.
        Verifies the record was actually removed from the database.
        """
        client.delete("/api/v1/users/me", headers=auth_headers)

        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": registered_user_payload["email"],
                "password": registered_user_payload["password"],
            }
        )
        assert response.status_code == 401

    def test_delete_account_requires_authentication(self, client):
        """Request without token must return 401."""
        response = client.delete("/api/v1/users/me")
        assert response.status_code == 401
