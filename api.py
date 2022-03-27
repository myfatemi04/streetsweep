import flask
import cv2

from recognize_from_video import get_class_likelihoods, get_imagenet_categories, get_object_bounding_boxes

app = flask.Flask(__name__)


upload_counter = 0
categories = get_imagenet_categories()
submissions = []


@app.route('/')
def index():
    return "You have reached the StreetSweep API endpoint."


@app.route("/submit_photo", methods=["POST"])
def submit_photo():
    global upload_counter

    request = flask.request
    if 'file' not in request.files:
        return "No file part"

    if 'lat' not in request.form or 'lng' not in request.form:
        return "Location was undefined"

    lat = request.form['lat']
    lng = request.form['lng']

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

    bounding_boxes = get_object_bounding_boxes(image)
    for ((x1, y1), (x2, y2)) in bounding_boxes:
        cropped_object = image[y1:y2, x1:x2]
        class_likelihoods = get_class_likelihoods(cropped_object)

    submissions.append({
        'lat': lat,
        'lng': lng,
        'class_likelihoods': class_likelihoods.tolist(),
    })

    return "Success"


@app.route("/submissions", methods=["GET"])
def get_submissions():
    return flask.jsonify(submissions)


app.run(port=5000, debug=True)
