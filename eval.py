import cv2
from tensorflow.keras.models import load_model
import numpy as np
model = load_model('BestModel_final.h5')
print('model loaded')
image = cv2.imread("D:\Final year project\\sample_wave.png")
# Get the dimensions of the image
height, width, channels = image.shape

# Print the dimensions
print(f"Width: {width}px")
print(f"Height: {height}px")
print(f"Channels: {channels}")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(gray_image, (224,224))
input_image = np.expand_dims(image, axis=0)
print('image processed')
predictions = model.predict(input_image)
print(predictions)

