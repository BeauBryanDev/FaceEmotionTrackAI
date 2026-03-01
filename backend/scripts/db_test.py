import sys
import os
import numpy as np
from sqlalchemy.orm import Session

# --- FIX PARA PASSLIB & BCRYPT ---
# Este parche es necesario en Python 3.12 para evitar el AttributeError
import logging
logging.getLogger("passlib").setLevel(logging.ERROR) 
# ---------------------------------

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.database import SessionLocal
from app.models.users import User
from app.core.security import get_password_hash

def test_biometric_storage():
    db: Session = SessionLocal()
    test_email = "vector_test_v2@example.com"
    
    try:
        print("--- Starting Biometric Storage Test ---")
        
        # Limpieza previa
        db.query(User).filter(User.email == test_email).delete()
        db.commit()

        # Generar vector 512D normalizado
        random_vector = np.random.rand(512).astype(np.float32)
        norm = np.linalg.norm(random_vector)
        normalized_vector = (random_vector / norm).tolist()

        # Crear usuario con contraseña corta (menor a 72 bytes)
        print("Creating test user with 512D embedding...")
        new_user = User(
            full_name="Vector Test User",
            email=test_email,
            # Usamos una contraseña simple para el test
            hashed_password=get_password_hash("secure_password"), 
            age=25,
            face_embedding=normalized_vector,
            is_active=True
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        print(f"✅ User saved successfully with ID: {new_user.id}")

        # Verificación de integridad
        retrieved_user = db.query(User).filter(User.email == test_email).first()
        if retrieved_user and retrieved_user.face_embedding is not None:
            print(f"✅ Success! Vector dimension in DB: {len(retrieved_user.face_embedding)}")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    test_biometric_storage()
    
    