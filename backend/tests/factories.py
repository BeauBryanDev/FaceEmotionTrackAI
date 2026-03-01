import factory
import factory.fuzzy
import factory.random

from app.models.users import User
from app.models.emotions import Emotion
from app.models.face_session import FaceSession


class UserFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Factory for generating test users."""

    class Meta:
        model = User
        sqlalchemy_session = None

    email = factory.Faker("email")
    full_name = factory.Faker("name")
    password = factory.Faker("password")
    age = factory.fuzzy.FuzzyInteger(0, 100)
    is_active = True
    
    
    @classmethod
    def _create(cls, model_class, *args, **kwargs):
        """Override the default _create() method to set the SQLAlchemy session."""
        kwargs["sqlalchemy_session"] = cls._meta.sqlalchemy_session
        return super()._create(model_class, *args, **kwargs)


class EmotionFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Factory for generating test emotion records."""

    class Meta:
        model = Emotion
        sqlalchemy_session = None

    user_id = factory.SelfAttribute('user.id')
    dominant_emotion = factory.fuzzy.FuzzyChoice(["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"])
    confidence = factory.fuzzy.FuzzyFloat(0.0, 1.0)
    emotion_scores = factory.Dict({"Anger": 0.05, "Contempt": 0.02, "Disgust": 0.03, "Fear": 0.01, "Happiness": 0.80, "Neutral": 0.05, "Sadness": 0.02, "Surprise": 0.02})
    timestamp = factory.Faker("date_time")
    
    
    @classmethod
    def _create(cls, model_class, *args, **kwargs):
        """Override the default _create() method to set the SQLAlchemy session."""
        kwargs["sqlalchemy_session"] = cls._meta.sqlalchemy_session
        return super()._create(model_class, *args, **kwargs)


class FaceSessionFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Factory for generating test face session records."""

    class Meta:
        model = FaceSession
        sqlalchemy_session = None

    user_id = factory.SelfAttribute('user.id')
    timestamp = factory.Faker("date_time")
    
    
    @classmethod
    def _create(cls, model_class, *args, **kwargs):
        """Override the default _create() method to set the SQLAlchemy session."""
        kwargs["sqlalchemy_session"] = cls._meta.sqlalchemy_session
        return super()._create(model_class, *args, **kwargs)
    
    
    def make_user(full_name="Test User", email="test@example.com"):
        return {"full_name": full_name, "email": email, "password": "Pass123", "age": 25}
