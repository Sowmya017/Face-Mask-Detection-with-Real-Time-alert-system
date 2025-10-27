# scripts/app_flask.py
from flask import Flask, Response, render_template_string
import cv2, time, numpy as np
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)
model = load_model("models/mask_detector.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

def gen():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x,y,w,h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224,224)).astype("float32")/255.0
            pred = model.predict(np.expand_dims(face_resized,0))[0][0]
            label = "Mask" if pred<0.5 else "No Mask"
            color = (0,255,0) if label=="Mask" else (0,0,255)
            cv2.rectangle(frame, (x,y),(x+w,y+h), color, 2)
            cv2.putText(frame, f"{label} {pred:.2f}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        ret2, jpg = cv2.imencode('.jpg', frame)
        frame_bytes = jpg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template_string('<html><body><h2 style= "text-align:center">Mask Stream</h2><div style="display: flex; justify-content: center; margin-top: 30px;"><img src="/video_feed" style="width: 60%; max-width: 700px; border-radius: 12px; box-shadow: 0 0 20px rgba(0,0,0,0.5);"></div></body></html>')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
