import cv2
import requests
import numpy as np
import tensorflow as tf
from datetime import datetime
from ultralytics import YOLO

# --- CONFIGURATION ---
# Replace with your actual Telegram Bot Token and Chat ID
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

# 1. LOAD MODELS (Methodology III)
# YOLO is used for identifying vehicles and road entities [cite: 688, 703]
# The paper mentions YOLOv8 as a high-performing choice [cite: 612]
yolo_model = YOLO('yolov8n.pt') 

# CNN is used for accident classification and severity analysis [cite: 689, 704]
# Labeling accidents as Minor, Moderate, or Severe [cite: 671]
def load_accident_cnn():
    # Placeholder for the CNN model described in the paper
    # Load your trained model here: tf.keras.models.load_model('accident_cnn.h5')
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(3, activation='softmax') 
    ])
    return model

cnn_model = load_accident_cnn()
severity_labels = ["Minor", "Moderate", "Severe"]

# --- FUNCTIONS ---

def send_telegram_alert(severity, frame):
    """Sends an immediate alert to emergency responders via Telegram API [cite: 579, 717]"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    location = "Coimbatore, India" # Placeholder location as per paper's origin [cite: 553]
    
    message = (f"🚨 ACCIDENT DETECTED! 🚨\n"
               f"Severity: {severity}\n"
               f"Time: {timestamp}\n"
               f"Location: {location}")
    
    # Send Text
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=payload)
    
    # Send Photo (Incident Evidence)
    _, img_encoded = cv2.imencode('.jpg', frame)
    photo_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    files = {'photo': ('incident.jpg', img_encoded.tobytes())}
    requests.post(photo_url, data={'chat_id': CHAT_ID}, files=files)

# --- MAIN DETECTION LOOP ---

# Start Video Feed (Step 1 of Algorithm 1) 
cap = cv2.VideoCapture(0) # 0 for local camera or CCTV stream link

print("System Initialized. Monitoring real-time traffic...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 2. OBJECT DETECTION (YOLO) - Step 2 [cite: 703]
    results = yolo_model(frame, stream=True, verbose=False)
    
    potential_accident = False
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            # Classes for vehicles: Car, Motorcycle, Bus, Truck
            if cls in [2, 3, 5, 7]:
                potential_accident = True # Logic to trigger CNN if objects are irregular [cite: 670]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 3. ACCIDENT CLASSIFICATION (CNN) - Step 3 [cite: 704]
    if potential_accident:
        # Preprocess frame for CNN classification [cite: 715]
        img_input = cv2.resize(frame, (224, 224))
        img_input = np.expand_dims(img_input / 255.0, axis=0)
        
        # Predict status and severity [cite: 690]
        prediction = cnn_model.predict(img_input, verbose=0)
        severity_idx = np.argmax(prediction)
        severity = severity_labels[severity_idx]
        
        # Simulation: For this demo, we check if prediction score is high
        if np.max(prediction) > 0.8: 
            # 4. ALERT GENERATION - Step 4 & 5 [cite: 707, 710]
            print(f"ALERT: {severity} Accident detected! Sending Telegram notification...")
            send_telegram_alert(severity, frame)
            
            # Visual Feedback on Screen
            cv2.putText(frame, f"CRITICAL: {severity} ACCIDENT", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display real-time feed [cite: 687]
    cv2.imshow("Hybrid Road Accident Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()