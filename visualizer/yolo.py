import os
import cv2
import matplotlib.pyplot as plt

def yolo_visualize(folder, image_name):
    image_path = os.path.join(folder + "/images", image_name)
    txt_path = os.path.join(folder + "/labels", image_name.replace(".jpg", ".txt"))

    image = cv2.imread(image_path)
    height, width, channel = image.shape # height, width, channel

    with open(txt_path, "r") as file:
        boxes = file.readlines() # list of string of boxes
        
    # get real values of boxes
    def get_values_from_string(line):
        line = line.strip().replace("\n", "").split(' ')
        x_center, y_center, width, height = float(line[1]), float(line[2]), \
                                            float(line[3]), float(line[4])
        return [x_center, y_center, width, height]

    real_boxes = [get_values_from_string(line) for line in boxes]
    for box in real_boxes:
        x_center, y_center, norm_width, norm_height = box
        x_left = x_center - norm_width/2; y_left = y_center - norm_height/2
        x_right = x_center + norm_width/2; y_right = y_center + norm_height/2
        
        image = cv2.rectangle(image,
                            (int(x_left*width), int(y_left*height)),
                            (int(x_right*width), int(y_right*height)),
                            color = (255, 0, 0),
                            thickness = 2)
    plt.imshow(image)