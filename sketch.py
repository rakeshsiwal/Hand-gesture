import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Create a white canvas to draw on
canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
# Variables to store previous coordinates for drawing smooth lines
prev_x, prev_y = None, None

# Process each frame to detect and draw hand landmarks
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    drawing = False  # Flag to indicate whether to draw or not

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(rgb_frame)

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the coordinates of the index finger tip
                h, w, c = frame.shape
                tipIndex_x, tipIndex_y = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
                tipThumb_x, tipThumb_y = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)

                # Draw on the canvas if the index finger is up
                if drawing:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas,(prev_x, prev_y),(tipIndex_x, tipIndex_y),(0, 0,255),5)

                tipIndexMid_y = int(hand_landmarks.landmark[6].y * h)
                tipIndex5_x, tipIndex5_y = int(hand_landmarks.landmark[5].x * w), int(hand_landmarks.landmark[5].y * h)

                if (tipIndex5_x < tipThumb_x+30) & (tipIndex_y < tipIndexMid_y)&(tipIndex5_y < tipThumb_y+30) & (tipIndex5_y > tipThumb_y-30)&(tipIndex5_x > tipThumb_x-30):
                    drawing = True
                else:
                    drawing = False
                    prev_x, prev_y = None, None

                # Update previous coordinates
                prev_x, prev_y = tipIndex_x, tipIndex_y

        # Combine the canvas and the frame
        combined = cv2.addWeighted(frame,0.5, canvas, 0.5, 0)

        # Display the combined image
        cv2.imshow('Finger Painting', combined)

        key=cv2.waitKey(1) & 0xFF

        # Break the loop when 'q' key is pressed
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Clear the canvas
            canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
