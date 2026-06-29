# SmartHR AI

An AI-powered Employee Attendance System that uses real-time face recognition and object detection to automatically mark employee attendance — no cards, no PINs, just your face.

---

## What It Does

When an employee walks in front of a camera, the system:

1. Detects a person using **YOLOv8** object detection
2. Locates their face using **Haar Cascade** face detection
3. Generates a **face embedding** using **DeepFace (Facenet)**
4. Compares it against all registered employees using **cosine similarity**
5. If matched — displays the employee's name, ID, department, and confidence score
6. **Marks attendance automatically** in the database (once per day per employee)

---

## Features

- **Live AI Terminal** — real-time face recognition via browser webcam
- **Employee Registration** — register employees with webcam capture or image upload
- **Face Validation** — rejects images with no face or multiple faces
- **Automatic Attendance** — marks attendance once per day per employee
- **Object Detection** — detects and labels other objects (phones, bags, etc.) in the frame
- **Attendance Log** — searchable and filterable attendance records
- **Export** — download attendance as CSV or Excel
- **Dashboard** — live stats, weekly attendance chart, department distribution
- **Settings** — adjustable recognition threshold, YOLO confidence, camera ID

---

## How It Works

### Registration Flow

```
User fills form  →  Webcam captures face  →  Haar Cascade validates face
      ↓
Image saved to database/  →  Employee record saved to SQLite
      ↓
DeepFace generates face embedding (Facenet)  →  Embedding cached in memory
```

### Recognition Flow

```
Browser webcam  →  Frame sent to /api/process_frame  →  YOLOv8 detects person
      ↓
Haar Cascade finds face in frame  →  DeepFace generates embedding for detected face
      ↓
Cosine similarity compared against all registered embeddings
      ↓
Match found (distance ≤ threshold)?
   YES → Display Name, ID, Department, Confidence → Mark attendance in SQLite
   NO  → Display UNKNOWN PERSON
```

### Embedding Cache

Face embeddings are stored **in memory** (not on disk). On every server start, embeddings are regenerated from the saved photos in the background — so the app is immediately available and loads silently behind the scenes.

When a new employee is registered, their embedding is added to the cache **without restarting** the server.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Flask |
| Face Recognition | DeepFace + Facenet model |
| Object Detection | YOLOv8n (Ultralytics) |
| Face Detection | OpenCV Haar Cascade |
| Database | SQLite |
| Image Processing | OpenCV (headless) |
| Frontend | HTML + CSS + Vanilla JS |
| Production Server | Gunicorn |
| Deployment | Railway (Docker) |

---

## Project Structure

```
SmartHR-AI/
│
├── app.py                  # Flask application — all routes and AI logic
│
├── templates/
│   ├── dashboard.html      # Stats, charts, recent activity
│   ├── register.html       # Employee registration with webcam
│   ├── recognition.html    # Live AI terminal
│   ├── attendance.html     # Attendance log with filters
│   └── settings.html       # System configuration
│
├── static/
│   ├── css/style.css       # Dark theme UI styles
│   └── js/app.js           # Dashboard charts and toast notifications
│
├── database/
│   ├── smarthr.db          # SQLite database (employees + attendance)
│   └── *.jpg               # Registered employee face photos
│
├── models/
│   └── yolov8n.pt          # YOLOv8 nano model weights
│
├── Dockerfile              # Python 3.11-slim container for Railway
├── Procfile                # Gunicorn start command
├── requirements.txt        # Python dependencies
└── railway.json            # Railway health check config
```

---

## Pages

### Dashboard `/`
Live overview of the system — total employees, registered faces, today's attendance count, system accuracy, a weekly attendance line chart, department distribution donut chart, and recent activity feed.

### Register Employee `/register`
Form to add a new employee. You can either upload a photo or capture one directly from your webcam. The system validates that the image contains exactly one face before saving. Once saved, the face embedding is generated immediately and recognition works without any server restart.

### Live AI Terminal `/recognition`
The main recognition screen. Click **Start Camera** to open your webcam. The browser sends frames to the server every second. The server runs YOLO + face detection + face recognition on each frame and returns the annotated image with bounding boxes. The right panel shows the current recognition status in real time.

### Attendance Log `/attendance`
Full attendance history with search by name or employee ID, filter by department, and filter by date. Export the filtered view as CSV or Excel.

### Settings `/settings`
Adjust the recognition threshold (how strict the face match must be), YOLO detection confidence, and camera device ID.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check — returns `{"status": "ok"}` |
| POST | `/api/register` | Register a new employee with face image |
| POST | `/api/process_frame` | Process a webcam frame — runs YOLO + face recognition |
| GET | `/api/live_status` | Current recognition status (name, ID, dept, confidence) |
| GET | `/api/attendance/list` | Attendance records with optional search/filter |
| GET | `/api/stats` | Dashboard stats (employee count, today's attendance, etc.) |
| GET | `/api/charts` | Weekly attendance and department data for charts |
| GET | `/api/export/csv` | Download attendance as CSV |
| GET | `/api/export/excel` | Download attendance as Excel |
| GET/POST | `/api/settings` | Get or update system settings |

---

## Running Locally

**Requirements:** Python 3.11, a webcam

```bash
# Clone the repo
git clone https://github.com/ani14006/SmartHR-AI.git
cd SmartHR-AI

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

Open `http://localhost:5001` in your browser.

> The first time you open the Live AI Terminal, DeepFace will download the Facenet model (~90 MB). This happens once and is cached automatically.

---

## Deploying to Railway

This project is pre-configured for Railway deployment using Docker.

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
3. Select this repository — Railway auto-detects the `Dockerfile`
4. Once deployed, go to **Settings → Networking → Generate Domain** to get your public URL

**Build time:** ~5–8 minutes (installs TensorFlow and dependencies)

### Optional Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FACE_MODEL` | `Facenet` | Face recognition model. Use `VGG-Face` for higher accuracy (needs 2 GB+ RAM) |
| `DATABASE_DIR` | `database` | Path to store the SQLite DB and face photos. Set to a Railway Volume path for persistence across deploys |
| `FLASK_DEBUG` | `0` | Set to `1` for debug mode (local development only) |

---

## Configuration

The recognition threshold controls how strict face matching is:

- **Lower value** (e.g. `0.30`) → stricter, fewer false positives, may miss some matches
- **Higher value** (e.g. `0.55`) → more lenient, recognizes more people, slight risk of false matches
- **Default: `0.40`** — works well for frontal face photos in reasonable lighting

Adjust this in the **Settings** page after deployment.

---

## Important Notes

- **Attendance is marked once per day** per employee — multiple recognitions in a day do not create duplicate records
- **Face images must contain exactly one face** — the system rejects uploads with no face or multiple faces
- **The SQLite database is ephemeral on Railway** by default — it resets on each redeploy. To make employee data persistent, mount a Railway Volume and set `DATABASE_DIR` to its path
- **The browser must have camera permission** for the Live AI Terminal to work. On deployed sites, HTTPS is required — Railway provides this automatically

---

## License

MIT License — see [LICENSE](LICENSE) for details.
