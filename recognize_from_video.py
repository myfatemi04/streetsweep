from torchvision import transforms
import torch
import time
import cv2
import numpy as np
from imread_from_url import imread_from_url
from PIL import Image

np.random.seed(0)
np.random.random(9)
np.random.random(15)
np.random.random(2021)

colors = np.random.randint(255, size=(100, 3), dtype=int)

try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter


class GenericDetector():

    def __init__(self, model_path, threshold=0.2):

        self.threshold = threshold

        # Initialize model
        self.model = self.initialize_model(model_path)

    def __call__(self, image):

        return self.detect_objects(image)

    def initialize_model(self, model_path):

        self.interpreter = Interpreter(model_path=model_path, num_threads=4)
        self.interpreter.allocate_tensors()

        # Get model info
        self.getModel_input_details()
        self.getModel_output_details()

    def detect_objects(self, image):

        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        self.inference(input_tensor)

        # Process output data
        detections = self.process_output()

        return detections

    def prepare_input(self, image):

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_tensor = cv2.resize(img, (self.input_width, self.input_height))
        input_tensor = input_tensor[np.newaxis, :, :, :]

        return input_tensor

    def inference(self, input_tensor):
        # Peform inference
        self.interpreter.set_tensor(
            self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()

    def process_output(self):

        # Get all output details
        boxes = self.get_output_tensor(0)
        classes = self.get_output_tensor(1)
        scores = self.get_output_tensor(2)
        num_objects = int(self.get_output_tensor(3))

        results = []
        for i in range(num_objects):
            if scores[i] >= self.threshold:
                result = {
                    'bounding_box': boxes[i],
                    'class_id': classes[i],
                    'score': scores[i]
                }
                results.append(result)
        return results

    def get_output_tensor(self, index):

        tensor = np.squeeze(self.interpreter.get_tensor(
            self.output_details[index]['index']))
        return tensor

    def getModel_input_details(self):

        self.input_details = self.interpreter.get_input_details()
        input_shape = self.input_details[0]['shape']
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.channels = input_shape[3]

    def getModel_output_details(self):
        self.output_details = self.interpreter.get_output_details()

    @staticmethod
    def draw_detections(image, detections, labels):

        img_height, img_width, _ = image.shape

        for idx, detection in enumerate(detections):
            box = detection['bounding_box']
            y1 = (img_height * box[0]).astype(int)
            y2 = (img_height * box[2]).astype(int)
            x1 = (img_width * box[1]).astype(int)
            x2 = (img_width * box[3]).astype(int)
            label = labels[idx]

            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (int(
                colors[idx, 0]), int(colors[idx, 1]), int(colors[idx, 2])), 5)

        return image


def get_object_bounding_boxes(frame):
    detections = detector(frame)

    img_height, img_width, _ = frame.shape

    bounding_boxes = []
    for detection in detections:
        box = detection['bounding_box']
        y1 = (img_height * box[0]).astype(int)
        y2 = (img_height * box[2]).astype(int)
        x1 = (img_width * box[1]).astype(int)
        x2 = (img_width * box[3]).astype(int)

        cropped = frame[y1:y2, x1:x2]
        if min(cropped.shape) > 0:
            bounding_boxes.append(((x1, y1), (x2, y2)))

    return bounding_boxes


def get_class_likelihoods(image):
    if type(image) == np.ndarray:
        image = Image.fromarray(image)
    elif type(image) == Image:
        pass
    else:
        print(f'Image type not supported: {type(image)}.')

    batch = torch.stack([preprocess(image)]).float()
    likelihoods = torch.softmax(model(batch), dim=1)

    return likelihoods


def detect_and_annotate(frame):  # , allowed_class_ids):
    detections = detector(frame)

    img_height, img_width, _ = frame.shape

    resized_generic_objects = []

    new_detections = []
    for idx, detection in enumerate(detections):
        box = detection['bounding_box']
        y1 = (img_height * box[0]).astype(int)
        y2 = (img_height * box[2]).astype(int)
        x1 = (img_width * box[1]).astype(int)
        x2 = (img_width * box[3]).astype(int)

        cropped = frame[y1:y2, x1:x2]
        if min(cropped.shape) > 0:
            resized = cv2.resize(cropped, (224, 224))
            resized_generic_objects.append(resized)
            new_detections.append(detection)

    detections = new_detections

    resized_generic_objects = torch.stack(
        [preprocess(Image.fromarray(t)) for t in resized_generic_objects]).float()

    likelihoods = torch.softmax(model(resized_generic_objects), dim=1)
    classification_ids = likelihoods.argmax(dim=1)
    labels = [categories[classification_ids[i]] +
              f"({max(likelihoods[i]) * 100:.2f}%)" for i in range(len(classification_ids))]

    detection_img = detector.draw_detections(
        frame, detections, labels)

    return detection_img


def get_imagenet_categories():
    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    return categories


model_path = 'object_detection_mobile_object_localizer_v1_1_default_1.tflite'
threshold = 0.2

detector = GenericDetector(model_path, threshold)

model = torch.hub.load('pytorch/vision:v0.10.0',
                       'resnet101', pretrained=True)
# Apparently, this is a really necessary line
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225]),
])


if __name__ == '__main__':

    categories = get_imagenet_categories()

    # image = Image.open("dog.jpeg")

    # likelihoods = torch.softmax(
    #     model(preprocess(image).unsqueeze(0)), dim=1)

    # top5_prob, top5_catid = torch.topk(likelihoods[0], 5)
    # for i in range(top5_prob.size(0)):
    #     print(categories[top5_catid[i]], top5_prob[i].item())

    # exit()

    # garbage_classes = [
    #     'beer bottle',
    #     'bottlecap',
    #     'beer glass',
    #     'water bottle',
    #     'plastic bag',
    # ]

    # garbage_class_ids = [categories.index(c) for c in garbage_classes]

    image = cv2.imread('img.png')

    cv2.imshow("Image", detect_and_annotate(image))  # , garbage_class_ids))
    cv2.waitKey(0)

    exit()

    cap = cv2.VideoCapture(0)

    with torch.no_grad():
        while True:
            _, frame = cap.read()

            detection_img = detect_and_annotate(frame)

            cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
            cv2.imshow("Detections", detection_img)

            if cv2.waitKey(1) == ord('q'):
                break
