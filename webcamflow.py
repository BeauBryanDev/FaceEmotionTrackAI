while True:
    ret, frame = cap.read()                     # ← OpenCV captura webcam
    
    faces = face_detector.detect(frame)         # Haar o DNN
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.equalizeHist(face_gray)   # arregla luz
        face_gray = cv2.resize(face_gray, (48, 48))
        
        face_tensor = transform(face_gray).unsqueeze(0).to(device)  # ToTensor + Normalize
        with torch.no_grad():
            output = model(face_tensor)
            emotion_idx = torch.argmax(output).item()
            emotion = emotions[emotion_idx]
        
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    cv2.imshow('Emotion Detector - Bryan Mode', frame)
