# Crop-and-Weed-Detection
This project addresses the problem of weed interference in agriculture, where weeds compete with crops for essential resources, reducing yield and quality. Traditional pesticide-based weed control can leave harmful residues on crops and harm the environment. To address this, we developed an intelligent weed detection system that uses targeted pesticide application, minimizing chemical exposure and optimizing usage.

Our model is based on YOLOv3, a high-performance deep learning model for real-time object detection, trained on a Kaggle dataset of 1,300 labeled images of sesame crops and weeds, each annotated with bounding boxes in YOLO format. To enhance accuracy, we used transfer learning with pretrained convolutional weights in the Darknet framework, leveraging Google Colab for efficient training.

For practical deployment, we integrated a Darknet-based inference system in Jupyter Notebook using OpenCV, allowing flexible field visualization. This solution promotes sustainable agriculture by reducing pesticide usage and supporting precision in crop management. Our work contributes to smart farming technology, enabling farmers to achieve healthier, high-quality crop production and laying the groundwork for future developments in automated agriculture.

For detection you need weights for network. Due to large file I am attaching google drive link. You have to download weight file unless you have your own weights file. https://drive.google.com/file/d/1-Aam2D-fqnwecbeHwa4rtzxtNjwcDkP6/view

You have to add weights flie into Crop_and_weed_detection > performing_detection > data > weights folder.
