import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

tf.compat.v1.disable_eager_execution()

module = hub.Module(
    "https://tfhub.dev/google/object_detection/mobile_object_localizer_v1/1")

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    frame_resized = cv2.resize(frame, (192, 192))

    features = module(np.array([frame_resized]), as_dict=True)
    detection_boxes = features["detection_boxes"]

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
