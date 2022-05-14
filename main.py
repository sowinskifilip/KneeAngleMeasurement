import math
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculateAngle(landmark1, landmark2, landmark3):
    # Get the required landmarks coordinates.
    x1, y1 = landmark1
    x2, y2 = landmark2
    x3, y3 = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360

    if angle > 180.0:
        angle = 360 - angle

    # Return the calculated angle.
    return angle

def getCoords(landmarks, left_knee, right_knee):
    for (i, j) in zip(range(24, 30, 2), range(23, 29, 2)):
        right_knee.append([landmarks[mp_pose.PoseLandmark(i).value].x,
                                     landmarks[mp_pose.PoseLandmark(i).value].y])
        left_knee.append([landmarks[mp_pose.PoseLandmark(j).value].x,
                                    landmarks[mp_pose.PoseLandmark(j).value].y])
    return

def displayVisualization(knee_landmarks, connections, image, results):
    # Calculate angle
    angle = calculateAngle(knee_landmarks[0], knee_landmarks[1], knee_landmarks[2])

    # Pose Detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, connections, landmark_drawing_spec=None)

    # Visualize angles
    image = cv2.putText(image, str(angle), tuple(np.multiply(knee_landmarks[1], [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def CameraCapture():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:

        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make Detections
            results = pose.process(image)

            # Recolor image back to BGR for rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            leg_to_analyze = 'right'
            right_knee_landmarks = []
            left_knee_landmarks = []
            leg_connections = ([(24, 26), (26, 28), (23, 25), (25, 27)])


            # Check if any landmarks are detected.
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get landmarks coordinates
                getCoords(landmarks, left_knee_landmarks, right_knee_landmarks)

                # Selected leg measurment and visualization
                if (leg_to_analyze == 'left'):
                    leg_connections = (leg_connections[2], leg_connections[3])
                    image = displayVisualization(left_knee_landmarks, leg_connections, image, results)
                elif (leg_to_analyze == 'right'):
                    leg_connections = (leg_connections[0], leg_connections[1])
                    image = displayVisualization(right_knee_landmarks, leg_connections, image, results)

            cv2.imshow('Knee angle measurment', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return


def main():
    CameraCapture()

    return

if __name__ == '__main__':
    main()

