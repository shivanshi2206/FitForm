For more details regarding testing and deployment refer FitFormAI.pdf

# FitForm

**FitForm** is a real-time exercise posture correction system built with Python, OpenCV, and MediaPipe. It analyzes body movements using webcam input in real time and provides instant feedback to ensure correct exercise form along with the repetition counter.

## Features

- Real-time detection and correction of posture
- Supports exercises: Bicep Curls, Squats, Side Leg Raises, and Planks
- Visual feedback to reduce risk of injury
- Web-based interface using Flask

## Prerequisites

Make sure you have the following installed:

- Python 3.x
- pip (Python package installer)
- flask
- opencv
- mediapipe
- numpy

## Project Structure

```
FitForm/
├── app.py
├── BicepCurl.py
├── Plank.py
├── SideLegRaise.py
├── Squat.py
├── templates/
│   ├── index.html
│   ├── bicep_curl.html
│   ├── plank.html
│   ├── side_leg_raise.html
│   └── squat.html
├── static/
│   └── style.css

```

## Running the App

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser and go to:**
   ```
   http://localhost:5000
   ```

## How to Use

- Launch the app and choose an exercise from the homepage.
- Allow webcam access when prompted.
- Follow the instructions and perform the exercise.
- Get real-time feedback on posture and repetitions.
