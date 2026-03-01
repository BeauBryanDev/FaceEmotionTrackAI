import pytest
from datetime import datetime, timezone
from app.models.emotions import Emotion


# Helper Insert Emotion Records into test DB

def _insert_emotion(db_session, user_id: int, dominant_emotion: str,
                    confidence: float, emotion_scores: dict = None):
    """
    Inserts a single Emotion record directly into the SQLite test DB.
    Bypasses the WebSocket stream to set up test data declaratively.
    """
    record = Emotion(
        user_id          = user_id,
        dominant_emotion = dominant_emotion,
        confidence       = confidence,
        emotion_scores   = emotion_scores,
        timestamp        = datetime.now(timezone.utc)
    )
    db_session.add(record)
    db_session.commit()
    return record


# GET /api/v1/emotions/history

class TestEmotionHistory:
    """Integration tests for GET /api/v1/emotions/history."""

    def test_history_returns_200_when_empty(
        self, client, registered_user, auth_headers
    ):
        """
        Authenticated user with no emotion records must get 200
        with an empty records list, not 404.
        """
        response = client.get(
            "/api/v1/emotions/history",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_records"] == 0
        assert data["records"] == []


    def test_history_requires_authentication(self, client, registered_user):
        """Request without token must return 401."""
        
        response = client.get("/api/v1/emotions/history")
        
        assert response.status_code == 401


    def test_history_returns_records_for_current_user(
        self, client, db_session, registered_user, auth_headers
    ):
        """
        Emotions inserted for the authenticated user must appear
        in the history response.
        """
        user_id = registered_user["id"]
        _insert_emotion(db_session, user_id, "Happiness", 0.92)
        _insert_emotion(db_session, user_id, "Neutral",   0.78)
        
        response = client.get(
            "/api/v1/emotions/history",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_records"] == 2
        assert len(data["records"]) == 2
        
        
    def test_history_response_contains_required_fields(
        self, client, db_session, registered_user, auth_headers
    ):
        """
        Each record in history must contain id, dominant_emotion,
        confidence, and timestamp.
        """
        _insert_emotion(db_session, registered_user["id"], "Anger", 0.85)

        response = client.get(
            "/api/v1/emotions/history",
            headers=auth_headers
        )
        record = response.json()["records"][0]
        assert "id"               in record
        assert "dominant_emotion" in record
        assert "confidence"       in record
        assert "timestamp"        in record
        
        
    def test_history_invalid_emotion_filter_returns_400(
        self, client, registered_user, auth_headers
    ):
        """
        Filtering by an emotion not in the 8 valid classes must return 400.
        """
        response = client.get(
            "/api/v1/emotions/history?emotion_filter=InvalidEmotion",
            headers=auth_headers
        )
        assert response.status_code == 400
        
        
    
    def test_history_valid_emotion_filter_returns_200(
        self, client, db_session, registered_user, auth_headers
    ):
        """
        Filtering by a valid emotion class must return 200
        with only records matching that emotion.
        """
        user_id = registered_user["id"]
        _insert_emotion(db_session, user_id, "Happiness", 0.92)
        _insert_emotion(db_session, user_id, "Sadness",   0.65)

        response = client.get(
            "/api/v1/emotions/history?emotion_filter=Happiness",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_records"] == 1
        assert data["records"][0]["dominant_emotion"] == "Happiness"
        
        
    def test_history_pagination_default_page_size(
        self, client, db_session, registered_user, auth_headers
    ):
        """
        Default page_size is 10. Inserting 15 records must return
        10 in the first page and total_records=15.
        """
        user_id = registered_user["id"]
        for i in range(15):
            _insert_emotion(db_session, user_id, "Neutral", 0.70)

        response = client.get(
            "/api/v1/emotions/history",
            headers=auth_headers
        )
        data = response.json()
        assert data["total_records"] == 15
        assert len(data["records"]) == 10
        
        
# GET /api/v1/emotions/summary

class TestEmotionSummary:
    """Integration tests for GET /api/v1/emotions/summary."""

    def test_summary_returns_200_when_empty(
        self, client, registered_user, auth_headers
    ):
        """
        User with no emotion records must get 200 with total_detections=0.
        """
        response = client.get(
            "/api/v1/emotions/summary",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_detections"] == 0
        assert data["dominant_emotion"] is None


    def test_summary_requires_authentication(self, client, registered_user):        
        """Request without token must return 401."""
        
        response = client.get("/api/v1/emotions/summary")
        
        assert response.status_code == 401


    def test_summary_returns_dominant_emotion(
        self, client, db_session, registered_user, auth_headers
    ):
        """
        Emotions inserted for the authenticated user must appear
        in the summary response.
        """
        user_id = registered_user["id"]
        _insert_emotion(db_session, user_id, "Happiness", 0.92)
        _insert_emotion(db_session, user_id, "Happiness", 0.88)
        _insert_emotion(db_session, user_id, "Neutral",   0.78)

        response = client.get(
            "/api/v1/emotions/summary",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_detections"] == 3
        assert data["dominant_emotion"] == "Happiness"
        
        
    def test_summary_requires_authentication(self, client):
        """Request without token must return 401."""
        response = client.get("/api/v1/emotions/summary")
        assert response.status_code == 401
        
        
    def test_summary_returns_correct_dominant_emotion(
        self, client, db_session, registered_user, auth_headers
    ):
        """
        The emotion with the most records must be returned as dominant_emotion.
        """
        user_id = registered_user["id"]
        _insert_emotion(db_session, user_id, "Happiness", 0.92)
        _insert_emotion(db_session, user_id, "Happiness", 0.88)
        _insert_emotion(db_session, user_id, "Neutral",   0.78)

        response = client.get(
            "/api/v1/emotions/summary",
            headers=auth_headers
        )
        data = response.json()
        assert data["dominant_emotion"] == "Happiness"
        assert data["total_detections"] == 3
        
        
    def test_summary_emotion_stats_contains_percentage(
        self, client, db_session, registered_user, auth_headers
    ):
        """
        Each entry in emotion_stats must contain emotion, count,
        percentage and avg_confidence.
        """
        user_id = registered_user["id"]
        _insert_emotion(db_session, user_id, "Neutral", 0.80)

        response = client.get(
            "/api/v1/emotions/summary",
            headers=auth_headers
        )
        stat = response.json()["emotion_stats"][0]
        assert "emotion"        in stat
        assert "count"          in stat
        assert "percentage"     in stat
        assert "avg_confidence" in stat
        
        
# GET /api/v1/emotions/scores


class TestEmotionScores:
    """Integration tests for GET /api/v1/emotions/scores."""

    def test_scores_returns_200_when_no_jsonb_data(
        self, client, registered_user, auth_headers
    ):
        """
        When no emotion records have emotion_scores populated,
        the endpoint must return 200 with count=0 and a message.
        """
        response = client.get(
            "/api/v1/emotions/scores",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0

    
    def test_scores_requires_authentication(self, client):
        """Request without token must return 401."""
        response = client.get("/api/v1/emotions/scores")
        assert response.status_code == 401


    def test_scores_returns_records_with_emotion_scores(
        self, client, db_session, registered_user, auth_headers
    ):
        """
        Records with emotion_scores populated must be returned by this endpoint.
        """
        user_id = registered_user["id"]
        scores = {
            "Anger": 0.05, "Contempt": 0.02, "Disgust": 0.03,
            "Fear": 0.01, "Happiness": 0.80, "Neutral": 0.05,
            "Sadness": 0.02, "Surprise": 0.02
        }
        _insert_emotion(
            db_session, user_id, "Happiness", 0.80,
            emotion_scores=scores
        )

        response = client.get(
            "/api/v1/emotions/scores",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["records"][0]["dominant_emotion"] == "Happiness"
        
        
        