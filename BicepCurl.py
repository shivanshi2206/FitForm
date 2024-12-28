import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

cap = cv2.VideoCapture(1)
counter = 0
stage = "down"
shoulder_angles = []
elbow_angles = []
torso_angles = []
sWin = 5 
up_frame_count = 0
down_frame_count = 0
hold_threshold = 3
# Create a named window and set it to always be on top
cv2.namedWindow("Curl Up", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Curl Up", cv2.WND_PROP_TOPMOST, 1)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image= cv2.resize(image, (1550, 780))
        cv2.rectangle(image,(0,0),(229,91),(212, 255, 223),-1)
        try:
            landmarks = results.pose_landmarks.landmark
            shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip =[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
            shoulder_angle=calculate_angle(hip,shoulder,elbow)
            elbow_angle=calculate_angle(shoulder,elbow,wrist)
            vertical_ref = [hip[0], hip[1] + 0.1] 
            torso_angle = calculate_angle(shoulder, hip, vertical_ref)

            cv2.putText(image, str(int(shoulder_angle)), tuple(np.multiply(shoulder, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(int(elbow_angle)), tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(int(torso_angle)), tuple(np.multiply(hip, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            ideal_shoulder_angle_range = (40, 100)
            ideal_elbow_angle_range = (40, 160)  
            ideal_torso_angle_range = (162, 180)  
            shoulder_angles = []
            elbow_angles = []
            torso_angles = []
            sWin = 5  
            shoulder_angles.append(shoulder_angle)
            elbow_angles.append(elbow_angle)
            torso_angles.append(torso_angle)
            if len(shoulder_angles) > sWin:
                shoulder_angles.pop(0)
                elbow_angles.pop(0)
                torso_angles.pop(0)
            shoulder_angle = np.mean(shoulder_angles)
            elbow_angle = np.mean(elbow_angles)
            torso_angle = np.mean(torso_angles)
            if shoulder_angle > 16:
                cv2.putText(image, "Bring elbow close", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            elif elbow_angle>30 and stage=="down":
                cv2.putText(image, "Raise your arm higher", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            elif elbow_angle<150 and stage=="up":
                 cv2.putText(image, "Lower your arm", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            if (shoulder_angle<16 and
                elbow_angle<30 and
                ideal_torso_angle_range[1] >= torso_angle >= ideal_torso_angle_range[0]):
                down_frame_count = 0
                up_frame_count += 1
                if up_frame_count >= hold_threshold and stage == "down":
                    stage = "up"
                    counter += 1
                    up_frame_count = 0

            elif (shoulder_angle<16 
            and ideal_torso_angle_range[1] >= torso_angle >= ideal_torso_angle_range[0]
            and elbow_angle>150):
                up_frame_count = 0
                down_frame_count += 1
                if down_frame_count >= hold_threshold and stage == "up":
                    stage = "down"
                    down_frame_count = 0  
        except:
            pass
        cv2.putText(image,"REPS",(15,17),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (15,65),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 195, 0), 2, cv2.LINE_AA)
        cv2.putText(image,"STAGE",(65,17),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (65,65),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 195, 0), 2, cv2.LINE_AA)
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)    
        cv2.imshow("Curl Up", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Curl Up", cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()