import math
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

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

    # Return the calculated angle.
    return angle

def CameraCapture():
    cap = cv2.VideoCapture(0)
    # Initiate holistic model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make Detections
            results = pose.process(image)
            # print(results.face_landmarks)

            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

            # Recolor image back to BGR for rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # # Draw face landmarks
            # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
            #
            # # Right hand
            # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            #
            # # Left Hand
            # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Raw Webcam Feed', image)

            right_leg_landmarks = []

            # Check if any landmarks are detected.
            if results.pose_landmarks:
                # Iterate over the detected landmarks.
                # for landmark in results.pose_landmarks.landmark:
                #     print(landmark)

                for i in range(24,30,2):
                    # Display the found landmarks after converting them into their original scale.

                    # print(f'{mp_pose.PoseLandmark(i).name}:')
                    # print(f'x: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x}')
                    # print(f'y: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y}')

                    # right_leg_landmarks.append({'x' : results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x,
                    #                             'y' : results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y})

                    right_leg_landmarks.append([results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x,
                                               results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y])

                angle = calculateAngle(right_leg_landmarks[0], right_leg_landmarks[1], right_leg_landmarks[2])
                print(angle)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return


def main():
    CameraCapture()

    # angle = calculateAngle((558, 326, 0), (642, 333, 0), (718, 321, 0))
    return

if __name__ == '__main__':
    main()

