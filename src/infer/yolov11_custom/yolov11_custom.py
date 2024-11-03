import tqdm
import os
import numpy as np
import albumentations as A
import argparse
from ultralytics import YOLO
import cv2

def infer_yolov11_custom(data_dir, weight_path, type_p=True):
    progress_bar = tqdm.tqdm(
        os.listdir(data_dir),
        total = len(os.listdir(data_dir))
    )
    ratio = 0
    gt_labels = [0, 1, 2, 3]
    model = YOLO(weight_path)
    
    with open("cus_predict.txt", "w") as file:
        for image_name in progress_bar:
            image_path = os.path.join(data_dir, image_name)
#           x_left, y_left, x_right, y_right = boxes[id]
#           box_width = x_right - x_left; box_height = y_right - y_left
    
#           x_center, y_center = (x_left + box_width/2) / width, (y_left + box_height/2) / height
#           file.write(f"{image_name} {int(labels[id])} {x_center} {y_center} {box_width/width} {box_height/height} {conf}\n")
       
    
            # load the image
            if type_p:
                det = model(image_path, imgsz=1280, conf=0.01, iou=0.45, verbose=False, augment=True)[0]
                conf_scores = det.boxes.conf.data.cpu().numpy()
                labels = det.boxes.cls.data.cpu().numpy()
                boxes = det.boxes.xywhn.data.cpu().numpy()
                height, width = det.orig_shape[0:2]
                            
                for id in range(len(boxes)):
                    conf = conf_scores[id]
                    file.write(f"{image_name} {int(labels[id])} {boxes[id][0]} {boxes[id][1]} {boxes[id][2]} {boxes[id][3]} {conf}\n")
            else:
                
                det = model(image_path, conf=0.01, iou=0.45, verbose=False, augment=True)[0]
                boxes = det.boxes.xyxy.data.cpu().numpy()
                height, width = det.orig_shape[0:2]

                for box in boxes:
                    predicted = box.astype(int)
                    sc_w, sc_h = predicted[2] - predicted[0], predicted[3] - predicted[1]
                    tmp_ratio = max(sc_w/width, sc_h/height)
                    if ratio < tmp_ratio:
                        ratio = tmp_ratio
                        if ratio > 0.5:
                            break

                if ratio > 0.5:
                    pad = 80
                    image = det.orig_img[:,:,::-1]
                    image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0) 
                    det = model(image, imgsz=1280, conf=0.01, iou=0.45, verbose=False, augment=True)[0]
                    conf_scores = det.boxes.conf.data.cpu().numpy()
                    labels = det.boxes.cls.data.cpu().numpy()
                    boxes = det.boxes.xyxy.data.cpu().numpy()
                    if boxes.shape[0] > 0:
                        boxes = boxes + pad

                    for id in range(len(boxes)):
                        xmin, ymin, xmax, ymax = boxes[id, :].astype(int)
                        box_width = xmax - xmin; box_height = ymax - ymin

                        xmin = min(max(0, xmin), width); xmax = min(max(0, xmax), width)
                        ymin = min(max(0, ymin), height); ymax = min(max(0, ymax), height)
                        x_c = (xmax - box_width / 2) / width; y_c = (ymax - box_height / 2) / height

                        conf = conf_scores[id]
                        file.write(f"{image_name} {int(labels[id])} {str(x_c)[:8]} {str(y_c)[:8]} {str(box_width / width)[:8]} {str(box_height / height)[:8]} {str(conf)[:6]}\n")

                else:
    #                 det = model(image_path, conf=0.01, iou=0.45, device="cpu", verbose=False)[0]
    #                 conf_scores = det.boxes.conf.data.cpu().numpy()
    #                 labels = det.boxes.cls.data.cpu().numpy()
    #                 boxes = det.boxes.xyxy.data.cpu().numpy()

                    det = model(image_path, imgsz=1280, conf=0.01, iou=0.45, verbose=False, augment=True)[0]
                    conf_scores = det.boxes.conf.data.cpu().numpy()
                    labels = det.boxes.cls.data.cpu().numpy()
                    boxes = det.boxes.xywhn.data.cpu().numpy()
                    height, width = det.orig_shape[0:2]

                    for id in range(len(boxes)):
                        conf = conf_scores[id]
                        file.write(f"{image_name} {int(labels[id])} {boxes[id][0]} {boxes[id][1]} {boxes[id][2]} {boxes[id][3]} {conf}\n")
#                 x_left, y_left, x_right, y_right = boxes[id]
#                 box_width = x_right - x_left; box_height = y_right - y_left
    
#                 x_center, y_center = (x_left + box_width/2) / width, (y_left + box_height/2) / height
#                 file.write(f"{image_name} {int(labels[id])} {x_center} {y_center} {box_width/width} {box_height/height} {conf}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference yolov11 custom")
    parser.add_argument(
        "--weigth_path",
        type=str,
        default="yolov11l.pt"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to test folder"
    )

    args = parser.parse_args()
    infer_yolov11_custom(args.weight_path, args.data_dir)