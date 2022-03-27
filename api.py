import time

import cv2
import flask
import numpy as np
from flask_cors import CORS, cross_origin

from recognize_from_video import (get_class_likelihoods,
                                  get_imagenet_categories,
                                  get_object_bounding_boxes)

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={'/submissions': {'origins': '*'}})

upload_counter = 0
categories = get_imagenet_categories()
submissions = []

colors = np.random.randint(255, size=(100, 3), dtype=int)


@app.route("/uploads/<path:path>")
def send_image(path):
    return flask.send_from_directory('uploads', path)


@app.route("/annotations/<path:path>")
def send_annotation(path):
    return flask.send_from_directory('annotations', path)


@app.route('/')
def index():
    return "You have reached the StreetSweep API endpoint."


@app.route("/submit_photo/<lat>,<lng>", methods=["POST"])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def submit_photo(lat, lng):
    global upload_counter

    request = flask.request
    if 'file' not in request.files:
        return "No file part"

    try:
        lat = float(lat)
        lng = float(lng)
    except:
        return "Location was NaN"

    import os

    if not os.path.exists('./uploads'):
        os.mkdir('./uploads')

    file = request.files['file']
    file.save(f'./uploads/{upload_counter}.jpg')

    upload_counter += 1

    # Perform inference.
    image = cv2.imread(f'./uploads/{upload_counter - 1}.jpg')

    all_likelihoods = []
    bounding_boxes = get_object_bounding_boxes(image)
    for idx, ((x1, y1), (x2, y2)) in enumerate(bounding_boxes):
        cropped_object = image[y1:y2, x1:x2]
        class_likelihoods = get_class_likelihoods(cropped_object)
        all_likelihoods.append(class_likelihoods)

        cv2.rectangle(image, (x1, y1), (x2, y2), (int(
            colors[idx, 0]), int(colors[idx, 1]), int(colors[idx, 2])), 5)

    # Save an annotated version of the image.
    cv2.imwrite(f'./annotations/{upload_counter - 1}.jpg', image)

    submissions.append({
        'id': upload_counter,
        'lat': lat,
        'lng': lng,
        'class_likelihoods': [L.tolist() for L in all_likelihoods],
        'timestamp': int(round(time.time() * 1000))  # milliseconds
    })

    return "Success"


@app.route("/submissions", methods=["GET"])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def get_submissions():
    return flask.jsonify(submissions)


app.run(port=5555, debug=True)
