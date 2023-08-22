from pathlib import Path
import tensorflow as tf
import numpy as np
import cv2 as cv
from PIL import Image

SINGLE_TEST = False
SHOW_WINDOWS = False

# Load your trained character recognition model
model = tf.keras.models.load_model('./models/TMNIST_model.keras')

# Load and preprocess the larger image
image_path = Path().absolute().joinpath('data', 'stromzaehler_c_e.png')
image = Image.open(str(image_path)).convert('L')

img = cv.imread(str(image_path))
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Define sliding window parameters
input_width = 28
input_height = 28
num_channels = 1
window_width = 72
window_height = 133
stride_x = 20
stride_y = 1
window_size = (window_width, window_height)
stride = (stride_x, stride_y)

Threshold = 0.98

#test single iamges:
if SINGLE_TEST:
    t_image = cv.imread(str(image_path))
    t_image = cv.cvtColor(t_image, cv.COLOR_BGR2GRAY)
    t_image = cv.resize(t_image, (28, 28))
    t_image = t_image.reshape(1, 28, 28, 1) / 255.0
    pred = model.predict(t_image)
    print(np.argmax(pred))
    np.set_printoptions(suppress=True)
    print(pred[0].astype(np.float64))

best_predictions = []

# Iterate through sliding windows
for y in range(0, image.height - window_size[1] + 1, stride[1]):
    for x in range(0, image.width - window_size[0] + 1, stride[0]):
        # Extract a window from the image
        window = img[y:y+window_size[1], x:x+window_size[0]]
        resized_window = cv.resize(window, (28, 28))
        normalized_window = resized_window.reshape(1, 28, 28, 1) / 255.0

        # Perform character recognition inference
        predictions = model.predict(normalized_window)
        max_prediction = np.max(predictions)
        prediction = np.argmax(predictions)

        if SHOW_WINDOWS:
            cv.imshow('image', resized_window)
            cv.waitKey(0)
            print(max_prediction)
            print(prediction)

        if max_prediction < Threshold or not np.argmax(predictions) in [0, 1, 4, 5, 7]:
            continue

        best_predictions.append((x, y, predictions))
        predicted_number = np.argmax(predictions)

        #print("Window at (x={}, y={}): Predicted Number: {}, Predictions: {}".format(x, y, predicted_number, predictions[0]))


# Draw bounding boxes around the detected characters converted to origial size
dp_image = cv.imread(str(image_path))

top_padding = 50
bottom_padding = 50
left_padding = 50
right_padding = 50

# Calculate the dimensions of the new canvas
new_height = dp_image.shape[0] + top_padding + bottom_padding
new_width = dp_image.shape[1] + left_padding + right_padding

# Create the new canvas filled with a background color (white in this case)
background_color = (255, 255, 255)  # White color in BGR format
padded_image = np.ones((new_height, new_width, 3), dtype=np.uint8) * background_color

# Place the original image onto the padded canvas
padded_image[top_padding:top_padding + dp_image.shape[0], left_padding:left_padding + dp_image.shape[1]] = dp_image
padded_image = padded_image.astype(np.uint8)

for x, y, predictions in best_predictions:
    x = x+left_padding
    y = y+top_padding
    ww = window_size[0]
    wh = window_size[1]

    cv.rectangle(padded_image, (x, y), (x + ww, y + wh), (0, 255, 0), 2)

    predicted_number = np.argmax(predictions)
    print("Window at (x={}, y={}): Predicted Number: {}, %: {}".format(x, y, predicted_number, predictions[0][predicted_number]))
    cv.putText(padded_image, str(predicted_number), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)




cv.imshow('image', padded_image)
cv.waitKey(0)