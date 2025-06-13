# 💤 Drowsiness Detection System

This is a simple web-based drowsiness detection system using **Flask** as the backend and **HTML5 camera**, **Bootstrap**, and **jQuery** for the frontend interface. The app captures frames from the user's webcam, analyzes them, and gives real-time feedback on drowsiness level.

---

## 📦 Installation

1. Open **Terminal** (Linux/Mac) or **Command Prompt** (Windows).

2. Navigate to the project directory:

   ```bash
   cd DrowsinessDetectionSystem
   ```

3. Install all required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run

1. Make sure you are using **Python 3**.

2. Run the Flask application:

   ```bash
   python app.py
   ```

3. After the server starts successfully, open your web browser and go to:

   ```
   http://127.0.0.1:5000
   ```

---

## 💡 Features

* Live webcam access through the browser (with user's permission).
* Circular camera UI with a colored ring:

  * **Gray**: Normal state.
  * **Yellow**: Warning (drowsy detected).
  * **Red**: Critical (highly drowsy).
* Captures **6 images per second**, sends **30 images every 5 seconds** to backend for processing.
* Asynchronous communication using **AJAX**.

---

## 📁 Project Structure

```
DrowsinessDetectionSystem/
├── app.py
├── templates/
│   └── index.html
├── static/
│   ├── lib/
│   │   ├── bootstrap.min.css
│   │   ├── bootstrap.min.js
│   │   ├── jquery.min.js
|   ├── warning.mp3
├── app.py
├── AttLayer.py
├── extract_features.py
├── LSTMAtt.py
└── README.md
└── requirements.txt
```

---

## 📌 Notes

* Make sure your browser allows webcam access.
* Test on modern browsers like **Chrome**, **Firefox**, or **Edge**.
* Adjust detection logic in the backend route `/DetectDrowiness` as needed.
