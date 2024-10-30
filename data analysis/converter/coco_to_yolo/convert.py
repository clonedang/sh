import shutil
import os
import json
from pycocotools.coco import COCO
import argparse

def main(cur_path, cur_json, img_path, new_path):
    coco = COCO(cur_json)
    imgIds = coco.getImgIds()

    for file in os.listdir(cur_path):
        shutil.copy(os.path.join(cur_path, file), img_path)

    for id in imgIds:
        img = coco.loadImgs(id)[0]
        img_name = img["file_name"]
        anns = coco.loadAnns(coco.getAnnIds(imgIds = id))
        
        img_height = img["height"]; img_width = img["width"]
        txt_img = img_name.replace(".jpg", ".txt")

        with open(os.path.join(new_path, txt_img), "w") as file:
            for ann in anns:
                cat_id = ann["category_id"]
                x_left, y_left, width, height = ann["bbox"]
                x_center, y_center = (x_left + width/2)/img_width, (y_left + height/2)/img_height

                file.write(f"{cat_id} {x_center} {y_center} {width/img_width} {height/img_height}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("COCO to YOLO format")
    parser.add_argument(
        "--cur_path",
        type=str,
        help="Current directory contains images",
    )
    parser.add_argument(
        "--cur_json",
        type=str,
        help="Current json file",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        help="New directory contains images",
    )
    parser.add_argument(
        "--new_path",
        type=str,
        help="New directory contains labels",
    )

    args = parser.parse_args()
    main(args.cur_path, args.cur_json, args.img_path, args.new_path)