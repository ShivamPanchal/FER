from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained model
my_model = tf.keras.models.load_model('./saved_model/')

def detect_emotion():
    # Open a connection to the video camera (0 represents the default camera, change it if you have multiple cameras)
    vid = cv2.VideoCapture(0)

    # Initialize variables to store previous emotion prediction
    prev_prediction = None
    prev_status = ""

    while True:
        # Capture the video frame by frame
        ret, frame = vid.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)

        # Initialize face_roi
        face_roi = None

        # Iterate over each detected face
        for x, y, w, h in faces:
            roi_color = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_roi = roi_color

        # Perform emotion recognition if a face is detected
        if face_roi is not None:
            # Convert face_roi to grayscale
            gray_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Resize and normalize the grayscale image
            image = cv2.resize(gray_face_roi, (48, 48))
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=-1)
            image = image / 255.0

            # Make prediction using the model
            val = my_model.predict(image)
            prediction_value = np.argmax(val[0])

            # Interpret the prediction and display the emotion
            emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
            status = emotions[prediction_value]

            # Update the displayed label only if the predicted emotion has changed
            if prediction_value != prev_prediction:
                prev_prediction = prediction_value
                prev_status = status

        # Display the resulting frame with title and emotion label
        title = ""
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        title_x = int((frame.shape[1] - title_size[0]) / 2)
        cv2.putText(frame, title, (title_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw text box at the bottom center
        text = prev_status if prev_status else "No face detected"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = int((frame.shape[1] - text_size[0]) / 2)
        text_y = frame.shape[0] - 20
        cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', rgb_frame)

        # Convert the frame to bytes
        frame_bytes = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    vid.release()

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
