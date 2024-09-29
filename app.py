import cv2
from flask import Flask, render_template, Response, jsonify
import RPi.GPIO as GPIO
import time
from datetime import datetime
import sqlite3
import threading
import numpy as np

app = Flask(__name__)

# Ultrasonic sensor setup
GPIO.setmode(GPIO.BCM)
TRIG = 23
ECHO = 24
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# Servo motor setup
SERVO_PIN = 17
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM frequency
servo.start(0)

# SQLite setup
DATABASE = 'distances.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS distances (timestamp TEXT, distance REAL)''')
    conn.commit()
    conn.close()

def log_distance(distance):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('INSERT INTO distances (timestamp, distance) VALUES (?, ?)', (timestamp, distance))
    conn.commit()
    conn.close()

def get_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start_time = time.time()
    stop_time = time.time()

    while GPIO.input(ECHO) == 0:
        start_time = time.time()

    while GPIO.input(ECHO) == 1:
        stop_time = time.time()

    time_elapsed = stop_time - start_time
    distance = (time_elapsed * 34300) / 2

    log_distance(distance)  # Log distance to database
    return distance

def rotate_servo(angle):
    duty_cycle = 2 + (angle / 18)
    GPIO.output(SERVO_PIN, True)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(1)
    GPIO.output(SERVO_PIN, False)
    servo.ChangeDutyCycle(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/distance')
def distance():
    dist = get_distance()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return jsonify(distance=dist, timestamp=timestamp)

@app.route('/history')
def history():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM distances ORDER BY timestamp DESC LIMIT 10')
    rows = cursor.fetchall()
    conn.close()
    return jsonify(rows)

def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Camera could not be opened.")
        return

    success, first_frame = camera.read()
    if not success:
        print("Error: Could not read from the camera.")
        return

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)
    
    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Could not read from the camera.")
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            delta_frame = cv2.absdiff(first_gray, gray)
            thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
            thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

            contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) < 500:  # Minimum contour area to be considered as an object
                    continue
                (x, y, w, h) = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Center the object in the frame
                cx = x + w // 2
                cy = y + h // 2
                fx = frame.shape[1] // 2
                fy = frame.shape[0] // 2
                dx = fx - cx
                dy = fy - cy
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def update_distance():
    while True:
        distance = get_distance()
        if distance < 20:
            rotate_servo(180)
            time.sleep(1)
            rotate_servo(0)
        time.sleep(2)

if __name__ == "__main__":
    init_db()
    distance_thread = threading.Thread(target=update_distance)
    distance_thread.daemon = True
    distance_thread.start()

    try:
        app.run(host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        GPIO.cleanup()
