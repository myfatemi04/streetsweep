import flask
from flask_cors import CORS, cross_origin
import cv2

from recognize_from_video import get_class_likelihoods, get_imagenet_categories, get_object_bounding_boxes

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={'/submissions': {'origins': '*'}})

upload_counter = 0
categories = get_imagenet_categories()
submissions = []


@app.route('/')
def index():
    return "You have reached the StreetSweep API endpoint."


@app.route("/submit_photo/<lat>,<lng>", methods=["POST"])
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
    for ((x1, y1), (x2, y2)) in bounding_boxes:
        cropped_object = image[y1:y2, x1:x2]
        class_likelihoods = get_class_likelihoods(cropped_object)
        all_likelihoods.append(class_likelihoods)

    submissions.append({
        'lat': lat,
        'lng': lng,
        'class_likelihoods': [L.tolist() for L in all_likelihoods],
    })

    return "Success"


@app.route("/submissions", methods=["GET"])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def get_submissions():
    return flask.jsonify(submissions)


app.run(port=5000, debug=True)
