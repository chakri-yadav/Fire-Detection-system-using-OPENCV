import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model('InceptionV3.h5')

# Open the video capture
video = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = video.read()

    if not ret:
        break

    # Convert the captured frame into RGB
    im = Image.fromarray(frame, 'RGB')

    # Resize the frame to 224x224 as the model was trained on this size
    im = im.resize((224, 224))
    img_array = image.img_to_array(im)
    img_array = np.expand_dims(img_array, axis=0) / 255

    # Predict probabilities for the frame
    probabilities = model.predict(img_array)[0]

    # Predict if there is fire in the frame
    prediction = np.argmax(probabilities)

    if prediction == 0:
        # Convert the frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        print("Fire probability:", probabilities[prediction])

    # Display the captured frame
    cv2.imshow("Capturing", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the video capture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
