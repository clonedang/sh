import os
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

def coco_visualize(folder, image_name, json_file):
    data = COCO(json_file)
    image_path = os.path.join(folder, image_name)

    anns = data.loadAnns(data.getAnnIds())
    imgs = data.loadImgs(data.getImgIds())

    for image in imgs:
        if image["file_name"] == image_name:
            image_id = image["id"]

    image = cv2.imread(image_path)
    height, width, channel = image.shape # height, width, channel

    real_boxes = [ann for ann in anns if ann["image_id"] == image_id]

    for box in real_boxes:
        x_left, y_left, width, height = box["bbox"]
        image = cv2.rectangle(image,
                            (int(x_left), int(y_left)),
                            (int(x_left + width), int(y_left + height)),
                            color = (255, 0, 0),
                            thickness = 2)
    plt.imshow(image)