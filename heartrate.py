import cv2
import numpy as np
from scipy.signal import butter, lfilter
import os
def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

# Set up the webcam
cap = cv2.VideoCapture(0)

opencv_dir = os.path.dirname(cv2.__file__)
haarcascade_path = os.path.join(opencv_dir, 'data', 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(haarcascade_path)


# Define the ROI (Region of Interest) for the forehead
roi_x, roi_y, roi_w, roi_h = 100, 100, 200, 200

# Define the buffer to store the green channel values
buffer_size = 100
buffer = np.zeros((buffer_size,))

# Define the heart rate calculation parameters
fs = 30.0  # sampling frequency (frames per second)
cutoff = 1.0  # cutoff frequency for the low-pass filter
order = 4  # order of the low-pass filter

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Extract the ROI (forehead) from the face
        roi = gray[y + roi_y:y + roi_y + roi_h, x + roi_x:x + roi_x + roi_w]

        # Calculate the mean green channel value in the ROI
        green_channel = cv2.split(roi)[0]
        mean_green = np.mean(green_channel)

        # Add the mean green value to the buffer
        buffer = np.roll(buffer, -1)
        buffer[-1] = mean_green

        # Calculate the heart rate
        signal = buffer - np.mean(buffer)
        signal = butter_lowpass_filter(signal, cutoff, fs, order)
        peaks = np.where((np.diff(np.sign(signal)) > 0))[0]
        heart_rate = 60.0 / np.mean(np.diff(peaks) / fs)

        # Display the heart rate
        cv2.putText(frame, f'Heart Rate: {heart_rate:.2f} bpm', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Heart Rate Detection', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()