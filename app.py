from flask import Flask, render_template, Response
import subprocess
import threading
import cv2


app = Flask(__name__)

# Pre-initialize the camera
def warm_up_camera():
    cap = cv2.VideoCapture(1)
    if cap.isOpened():
        print("Camera warmed up successfully.")
        cap.release()
    else:
        print("Failed to warm up the camera.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/side_leg_raise")
def side_leg_raise():
    return render_template("side_leg_raise.html")
    
@app.route("/plank")
def plank():
    return render_template("plank.html")

@app.route("/squat")
def squats():
    return render_template("squat.html")  # Ensure this template exists

@app.route("/bicep_curl")
def bicep_curl():
    return render_template("bicep_curl.html")  # Ensure this template exists

import os

@app.route("/start_camera/<exercise>")
def start_camera(exercise):
    script_map = {
        "side_leg_raise": r"C:\Users\MY\Documents\capstone\SideLegRaise.py",
        "squats": r"C:\Users\MY\Documents\capstone\Squat.py",
        "bicep_curl": r"C:\Users\MY\Documents\capstone\BicepCurl.py",
        "plank": r"C:\Users\MY\Documents\capstone\Plank.py",
    }
    script_path = script_map.get(exercise)
    print(f"Exercise: {exercise}, Script Path: {script_path}")

    
    if script_path and os.path.exists(script_path):
        subprocess.Popen(["python", script_path], shell=True)
        return f"{exercise.capitalize()} module started successfully."
    else:
        return f"Error: {script_path or 'Script'} not found.", 404


if __name__ == "__main__":
    # Warm up the camera before starting the Flask app
    threading.Thread(target=warm_up_camera).start()
    app.run(debug=True)
