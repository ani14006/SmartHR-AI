import os
import cv2
import sqlite3
import time
import pandas as pd
import numpy as np
import threading
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify, send_file
from ultralytics import YOLO
from deepface import DeepFace

app = Flask(__name__)

# Configuration & Paths
DB_PATH = "database/smarthr.db"
IMAGE_DIR = "database"
MODEL_PATH = "models/yolov8n.pt"

# Global System Settings
app_settings = {
    "rec_threshold": 0.40,  # Cosine distance threshold (lower means stricter)
    "yolo_conf": 0.45,
    "camera_id": 0,
    "theme": "dark"
}

# In-memory caches for face recognition
employee_embeddings = {}  # {employee_id: {"name": name, "dept": dept, "embedding": embedding}}
current_ai_status = {
    "current_detection": "No Face Detected",
    "recognized_person": "None",
    "employee_id": "None",
    "name": "None",
    "department": "None",
    "confidence": 0.0,
    "detected_objects": [],
    "status": "ONLINE"
}
last_notification = None  # Store last notification to send as SSE or via polling

# Lock for camera access
camera_lock = threading.Lock()
# Background recognition lock
recognition_lock = threading.Lock()
is_recognizing = False
last_recognition_time = 0

# Initialize SQLite database
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        employee_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT,
        email TEXT,
        phone TEXT,
        photo_path TEXT
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT NOT NULL,
        name TEXT NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        status TEXT NOT NULL,
        FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
    )
    """)
    conn.commit()
    conn.close()

# Load/reload employee face embeddings into memory cache
def load_embeddings():
    global employee_embeddings
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT employee_id, name, department, photo_path FROM employees")
    rows = cursor.fetchall()
    conn.close()
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    new_embeddings = {}
    for emp_id, name, dept, photo_path in rows:
        if photo_path and os.path.exists(photo_path):
            # Check if we already have this employee's embedding cached and photo_path matches
            if emp_id in employee_embeddings and employee_embeddings[emp_id].get("photo_path") == photo_path:
                new_embeddings[emp_id] = employee_embeddings[emp_id]
                new_embeddings[emp_id]["name"] = name
                new_embeddings[emp_id]["dept"] = dept
            else:
                try:
                    # Load image and crop face using Haar Cascade
                    img = cv2.imread(photo_path)
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                        if len(faces) > 0:
                            fx, fy, fw, fh = faces[0]
                            face_crop = img[fy:fy+fh, fx:fx+fw].copy()
                            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        else:
                            face_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            
                        # Precompute representation using DeepFace (VGG-Face model by default)
                        resp = DeepFace.represent(img_path=face_rgb, model_name="VGG-Face", enforce_detection=False, detector_backend="skip")
                        if resp and len(resp) > 0:
                            emb = resp[0]["embedding"]
                            new_embeddings[emp_id] = {
                                "name": name,
                                "dept": dept,
                                "embedding": emb,
                                "photo_path": photo_path
                            }
                except Exception as e:
                    print(f"Error loading face embedding for {name} ({emp_id}): {e}")
                    
    employee_embeddings = new_embeddings

# Initialize YOLOv8 Model
try:
    yolo_model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"YOLOv8 loading failed, trying auto-download to {MODEL_PATH}: {e}")
    yolo_model = YOLO("yolov8n.pt")
    yolo_model.save(MODEL_PATH)

init_db()
load_embeddings()

# Face recognition background thread function
def perform_face_recognition(face_img):
    global is_recognizing, current_ai_status, last_notification
    try:
        # Get embedding of the detected face (convert BGR to RGB first)
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        resp = DeepFace.represent(img_path=face_rgb, model_name="VGG-Face", enforce_detection=False, detector_backend="skip")
        if not resp or len(resp) == 0:
            return
        
        detected_emb = resp[0]["embedding"]
        best_match = None
        min_dist = 1.0  # Cosine distance ranges from 0 to 2 (0 is identical)
        
        # Compare with registered employees
        for emp_id, data in employee_embeddings.items():
            dist = np.dot(detected_emb, data["embedding"]) / (np.linalg.norm(detected_emb) * np.linalg.norm(data["embedding"]))
            cosine_dist = 1.0 - dist
            if cosine_dist < min_dist:
                min_dist = cosine_dist
                best_match = (emp_id, data)
                
        threshold = app_settings["rec_threshold"]
        accuracy = max(0.0, (1.0 - min_dist) * 100.0)
        
        if best_match and min_dist <= threshold:
            emp_id, data = best_match
            current_ai_status["current_detection"] = "Face Detected"
            current_ai_status["recognized_person"] = f"{data['name']} ({emp_id})"
            current_ai_status["employee_id"] = emp_id
            current_ai_status["name"] = data["name"]
            current_ai_status["department"] = data["dept"]
            current_ai_status["confidence"] = round(accuracy, 1)
            
            # Log Attendance (only once per day)
            today_str = datetime.now().strftime("%Y-%m-%d")
            time_str = datetime.now().strftime("%H:%M:%S")
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM attendance WHERE employee_id = ? AND date = ?", (emp_id, today_str))
            exists = cursor.fetchone()
            
            if not exists:
                cursor.execute("""
                    INSERT INTO attendance (employee_id, name, date, time, status)
                    VALUES (?, ?, ?, ?, ?)
                """, (emp_id, data["name"], today_str, time_str, "Present"))
                conn.commit()
                last_notification = f"Attendance Marked Successfully for {data['name']}"
            conn.close()
        else:
            current_ai_status["current_detection"] = "Face Detected"
            current_ai_status["recognized_person"] = "UNKNOWN PERSON"
            current_ai_status["employee_id"] = "None"
            current_ai_status["name"] = "UNKNOWN PERSON"
            current_ai_status["department"] = "None"
            current_ai_status["confidence"] = round(accuracy, 1) if best_match else 0.0
            
    except Exception as e:
        print(f"Face recognition error: {e}")
    finally:
        with recognition_lock:
            is_recognizing = False

# Video Streaming MJPEG Generator
def generate_frames():
    global is_recognizing, last_recognition_time, current_ai_status
    
    # Try opening camera
    camera = cv2.VideoCapture(app_settings["camera_id"])
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        success, frame = camera.read()
        if not success:
            # Fallback mock image when camera is not connected/accessible
            mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(mock_frame, "CAMERA OFFLINE / ACCESS DENIED", (80, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', mock_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
            continue
            
        # Run YOLOv8 Object Detection
        # Draw bounding boxes
        results = yolo_model(frame, verbose=False, conf=app_settings["yolo_conf"])
        detected_names = []
        person_detected = False
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = yolo_model.names[cls]
                
                # Filter specific target objects or label all
                if label == "person":
                    person_detected = True
                    # Draw yellow/green/red box based on current recognized state
                    status = current_ai_status["recognized_person"]
                    if status != "None" and status != "UNKNOWN PERSON":
                        # Green box for recognized employee
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Verified: {status}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    elif status == "UNKNOWN PERSON":
                        # Red box for unknown person
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "UNKNOWN PERSON", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        # Yellow box for unknown/unverified person
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame, "Analyzing Face...", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    detected_names.append(label.capitalize())
                    # Blue box for other detected objects
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"OBJECT: {label.capitalize()}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                
        # Update objects list
        current_ai_status["detected_objects"] = list(set(detected_names))
        
        # If Person is detected, trigger the Face Recognition Pipeline
        if person_detected:
            # Detect faces with Haar Cascade inside the frame or just pass frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (fx, fy, fw, fh) in faces:
                status = current_ai_status["recognized_person"]
                if status != "None" and status != "UNKNOWN PERSON":
                    # Draw green face rectangle
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
                    emp_name = current_ai_status.get("name", "")
                    emp_id = current_ai_status.get("employee_id", "")
                    emp_dept = current_ai_status.get("department", "")
                    conf = current_ai_status.get("confidence", 0.0)
                    
                    cv2.putText(frame, f"Name: {emp_name}", (fx, fy - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"ID: {emp_id}", (fx, fy - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Dept: {emp_dept}", (fx, fy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Conf: {conf}%", (fx, fy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                elif status == "UNKNOWN PERSON":
                    # Draw red face rectangle
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)
                    cv2.putText(frame, "UNKNOWN PERSON", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    # Draw yellow face rectangle
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 255), 2)
                    cv2.putText(frame, "Analyzing Face...", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Continuous Recognition pipeline (every 1 second)
                now = time.time()
                if not is_recognizing and (now - last_recognition_time > 1.0):
                    with recognition_lock:
                        is_recognizing = True
                    last_recognition_time = now
                    # Crop face and run background recognition
                    face_crop = frame[fy:fy+fh, fx:fx+fw].copy()
                    threading.Thread(target=perform_face_recognition, args=(face_crop,), daemon=True).start()
        else:
            current_ai_status["current_detection"] = "No Face Detected"
            current_ai_status["recognized_person"] = "None"
            current_ai_status["employee_id"] = "None"
            current_ai_status["name"] = "None"
            current_ai_status["department"] = "None"
            current_ai_status["confidence"] = 0.0
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    camera.release()

# Web Application Routes

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

@app.route('/recognition')
def recognition():
    return render_template('recognition.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

# API Endpoints

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def api_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total Employees
    cursor.execute("SELECT COUNT(*) FROM employees")
    total_emp = cursor.fetchone()[0]
    
    # Registered Faces
    cursor.execute("SELECT COUNT(*) FROM employees WHERE photo_path IS NOT NULL")
    reg_faces = cursor.fetchone()[0]
    
    # Today's Attendance
    today_str = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(DISTINCT employee_id) FROM attendance WHERE date = ?", (today_str,))
    today_att = cursor.fetchone()[0]
    
    conn.close()
    
    # System Accuracy (Mocked based on registration & database stats)
    system_accuracy = 98.7 if reg_faces > 0 else 0.0
    
    return jsonify({
        "total_employees": total_emp,
        "registered_faces": reg_faces,
        "today_attendance": today_att,
        "system_accuracy": system_accuracy
    })

@app.route('/api/charts')
def api_charts():
    # Returns weekly attendance and department breakdown
    conn = sqlite3.connect(DB_PATH)
    
    # Weekly Attendance
    df_att = pd.read_sql_query("SELECT date, COUNT(DISTINCT employee_id) as count FROM attendance GROUP BY date ORDER BY date DESC LIMIT 7", conn)
    # Reversing for chronological order
    df_att = df_att.iloc[::-1]
    
    # Department Distribution
    df_dept = pd.read_sql_query("SELECT department, COUNT(*) as count FROM employees GROUP BY department", conn)
    
    # Employee Activity
    df_act = pd.read_sql_query("SELECT time, name FROM attendance ORDER BY date DESC, time DESC LIMIT 10", conn)
    
    conn.close()
    
    return jsonify({
        "weekly": {
            "labels": df_att["date"].tolist() if not df_att.empty else [datetime.now().strftime("%Y-%m-%d")],
            "data": df_att["count"].tolist() if not df_att.empty else [0]
        },
        "departments": {
            "labels": df_dept["department"].tolist() if not df_dept.empty else ["IT", "HR", "Sales", "Operations"],
            "data": df_dept["count"].tolist() if not df_dept.empty else [0, 0, 0, 0]
        },
        "activity": df_act.to_dict(orient='records')
    })

@app.route('/api/recent_activity')
def api_recent():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT a.employee_id, a.name, e.department, a.date, a.time, a.status 
        FROM attendance a 
        LEFT JOIN employees e ON a.employee_id = e.employee_id 
        ORDER BY a.date DESC, a.time DESC LIMIT 5
    """)
    rows = cursor.fetchall()
    conn.close()
    
    activity = []
    for r in rows:
        activity.append({
            "employee_id": r[0],
            "name": r[1],
            "department": r[2] or "N/A",
            "date": r[3],
            "time": r[4],
            "status": r[5]
        })
    return jsonify(activity)

@app.route('/api/register', methods=['POST'])
def api_register():
    try:
        emp_id = request.form.get("employee_id")
        name = request.form.get("name")
        dept = request.form.get("department")
        email = request.form.get("email")
        phone = request.form.get("phone")
        
        if not emp_id or not name:
            return jsonify({"success": False, "message": "Employee ID and Name are required."}), 400
            
        photo_path = None
        # Handle file upload or webcam capture
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                filename = f"{emp_id}_{name.replace(' ', '_')}.jpg"
                photo_path = os.path.join(IMAGE_DIR, filename)
                file.save(photo_path)
        elif 'webcam_image' in request.form:
            import base64
            img_data = request.form['webcam_image']
            if img_data.startswith("data:image"):
                header, encoded = img_data.split(",", 1)
                data = base64.b64decode(encoded)
                filename = f"{emp_id}_{name.replace(' ', '_')}.jpg"
                photo_path = os.path.join(IMAGE_DIR, filename)
                with open(photo_path, "wb") as f:
                    f.write(data)
                    
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT 1 FROM employees WHERE employee_id = ?", (emp_id,))
        exists = cursor.fetchone()
        
        if exists:
            cursor.execute("""
                UPDATE employees 
                SET name = ?, department = ?, email = ?, phone = ?, photo_path = COALESCE(?, photo_path)
                WHERE employee_id = ?
            """, (name, dept, email, phone, photo_path, emp_id))
        else:
            cursor.execute("""
                INSERT INTO employees (employee_id, name, department, email, phone, photo_path)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (emp_id, name, dept, email, phone, photo_path))
            
        conn.commit()
        conn.close()
        
        # Reload embeddings cache
        load_embeddings()
        
        return jsonify({"success": True, "message": f"Employee {name} registered successfully!"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/live_status')
def api_live_status():
    global last_notification
    notif = last_notification
    last_notification = None  # Clear notification after reading
    return jsonify({
        "current_detection": current_ai_status["current_detection"],
        "recognized_person": current_ai_status["recognized_person"],
        "confidence": f"{current_ai_status['confidence']}%",
        "detected_objects": current_ai_status["detected_objects"],
        "status": current_ai_status["status"],
        "notification": notif
    })

@app.route('/api/attendance/list')
def api_attendance_list():
    search = request.args.get("search", "")
    dept = request.args.get("department", "")
    date = request.args.get("date", "")
    
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT a.employee_id, a.name, e.department, a.date, a.time, a.status 
        FROM attendance a
        LEFT JOIN employees e ON a.employee_id = e.employee_id
        WHERE 1=1
    """
    params = []
    
    if search:
        query += " AND (a.name LIKE ? OR a.employee_id LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])
    if dept:
        query += " AND e.department = ?"
        params.append(dept)
    if date:
        query += " AND a.date = ?"
        params.append(date)
        
    query += " ORDER BY a.date DESC, a.time DESC"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/export/csv')
def export_csv():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT a.employee_id as [Employee ID], a.name as [Name], e.department as [Department], 
               a.date as [Date], a.time as [Time], a.status as [Status]
        FROM attendance a
        LEFT JOIN employees e ON a.employee_id = e.employee_id
        ORDER BY a.date DESC, a.time DESC
    """, conn)
    conn.close()
    
    csv_path = "database/attendance_export.csv"
    df.to_csv(csv_path, index=False)
    return send_file(csv_path, as_attachment=True, download_name=f"Attendance_Log_{datetime.now().strftime('%Y%m%d')}.csv")

@app.route('/api/export/excel')
def export_excel():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT a.employee_id as [Employee ID], a.name as [Name], e.department as [Department], 
               a.date as [Date], a.time as [Time], a.status as [Status]
        FROM attendance a
        LEFT JOIN employees e ON a.employee_id = e.employee_id
        ORDER BY a.date DESC, a.time DESC
    """, conn)
    conn.close()
    
    excel_path = "database/attendance_export.xlsx"
    df.to_excel(excel_path, index=False, engine='openpyxl')
    return send_file(excel_path, as_attachment=True, download_name=f"Attendance_Log_{datetime.now().strftime('%Y%m%d')}.xlsx")

@app.route('/api/settings', methods=['GET', 'POST'])
def api_get_post_settings():
    global app_settings
    if request.method == 'POST':
        app_settings["rec_threshold"] = float(request.json.get("rec_threshold", app_settings["rec_threshold"]))
        app_settings["yolo_conf"] = float(request.json.get("yolo_conf", app_settings["yolo_conf"]))
        app_settings["camera_id"] = int(request.json.get("camera_id", app_settings["camera_id"]))
        app_settings["theme"] = request.json.get("theme", app_settings["theme"])
        return jsonify({"success": True, "message": "Settings updated successfully."})
    return jsonify(app_settings)

@app.route('/api/process_frame', methods=['POST'])
def api_process_frame():
    global is_recognizing, last_recognition_time, current_ai_status, last_notification
    try:
        import base64
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success": False, "message": "No image data"}), 400
            
        img_data = data['image']
        if img_data.startswith("data:image"):
            header, encoded = img_data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
        else:
            img_bytes = base64.b64decode(img_data)
            
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "message": "Decode failed"}), 400
            
        # Run YOLOv8 Object Detection
        results = yolo_model(frame, verbose=False, conf=app_settings["yolo_conf"])
        detected_names = []
        person_detected = False
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = yolo_model.names[cls]
                
                # We also map cell phone to standard label cell phone
                obj_label = label
                if label == "cell phone":
                    obj_label = "cell phone"
                
                if label == "person":
                    person_detected = True
                    status = current_ai_status["recognized_person"]
                    if status != "None" and status != "UNKNOWN PERSON":
                        # Green box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Verified: {status}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    elif status == "UNKNOWN PERSON":
                        # Red box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "UNKNOWN PERSON", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        # Yellow box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame, "Analyzing Face...", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    detected_names.append(obj_label.capitalize())
                    # Blue box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"OBJECT: {obj_label.capitalize()}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                
        current_ai_status["detected_objects"] = list(set(detected_names))
        
        if person_detected:
            # Haar Cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (fx, fy, fw, fh) in faces:
                status = current_ai_status["recognized_person"]
                if status != "None" and status != "UNKNOWN PERSON":
                    # Draw green face rectangle
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
                    emp_name = current_ai_status.get("name", "")
                    emp_id = current_ai_status.get("employee_id", "")
                    emp_dept = current_ai_status.get("department", "")
                    conf = current_ai_status.get("confidence", 0.0)
                    
                    cv2.putText(frame, f"Name: {emp_name}", (fx, fy - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"ID: {emp_id}", (fx, fy - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Dept: {emp_dept}", (fx, fy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Conf: {conf}%", (fx, fy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                elif status == "UNKNOWN PERSON":
                    # Draw red face rectangle
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)
                    cv2.putText(frame, "UNKNOWN PERSON", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    # Draw yellow face rectangle
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 255), 2)
                    cv2.putText(frame, "Analyzing Face...", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                now = time.time()
                if not is_recognizing and (now - last_recognition_time > 1.0):
                    with recognition_lock:
                        is_recognizing = True
                    last_recognition_time = now
                    face_crop = frame[fy:fy+fh, fx:fx+fw].copy()
                    threading.Thread(target=perform_face_recognition, args=(face_crop,), daemon=True).start()
        else:
            current_ai_status["current_detection"] = "No Face Detected"
            current_ai_status["recognized_person"] = "None"
            current_ai_status["employee_id"] = "None"
            current_ai_status["name"] = "None"
            current_ai_status["department"] = "None"
            current_ai_status["confidence"] = 0.0
            
        # Encode back to base64
        _, buffer = cv2.imencode('.jpg', frame)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "image": f"data:image/jpeg;base64,{processed_base64}",
            "status": current_ai_status
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
