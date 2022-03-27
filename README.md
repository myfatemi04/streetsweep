# streetsweep

## Inspiration
We decided that there was too much trash in the street, but that it was also difficult to quantify how much trash was littered and use that information to effectively clean it up.

## What it does
StreetSweep uses computer vision and machine learning to allow users to submit pictures of areas with trash and display the location on a map. The images can come from anywhere - even pictures taken from fast-moving vehicles. Anyone who visits the website is able to see a public board of where trash is in their area. Our hope is that this enables concentrated efforts to reduce the amount of trash on the streets through the increased information flow.

## How we built it
When a user submits an image from the React frontend, it is sent as part of a POST request to a Flask API and stored as a temporary file. Then, the image is read by OpenCV, and we use a PyTorch model pre-trained on ImageNet to detect potential garbage. We persist the images and their object detections to the disk. As a final step in our data processing pipeline, we store an annotated version of the image and make it available through the Flask API, so when a user views which locations have trash on the frontend, they will be able to access the image that was submitted as well as the garbage that was detected by the AI.

The location and types of trash are then stored and displayed using a heatmap. The user can view the total number of trash submissions and the composition of the reports of the types of trash depending on which part of the map they view.

## Challenges we ran into
We had trouble finding a well-annotated dataset on garbage. The closest thing to what we needed only had 10 well-annotated classes. We decided to supplement our original dataset with ImageNet, a model with thousands of classes. However, we could find no public object detection models with the versatility of ImageNet. So, we used a generic object detection network to detect candidate objects, and then classified those candidate objects with ImageNet to mimic the ability to perform object segmentation at a fraction of the compute time and without needing to train another model.

## What's next for StreetSweep
In the future, we plan to expand StreetSweep by using local drone surveillance to mark down areas where trash is present. We also want to use a better-annotated dataset or create one ourselves. Potentially, the images collected by StreetSweep contributors could be used as a rolling dataset and be annotated by human volunteers to enable StreetSweep’s garbage detection recall and accuracy to improve over time.
