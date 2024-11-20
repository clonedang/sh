from pycocotools.coco import COCO
import json
import os
from tqdm import tqdm
from PIL import Image
from npencoder import NpEncoder
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

class ABR(object):
    def __init__(self, 
                 images_dir,
                 ann_file,
                 buffered_images_dir,
                 night_time = True):
        
        self.classes = [0, 1, 2, 3]
        self.images_dir = images_dir
        self.ann_file = ann_file
        self.night_time = night_time
        self.buffered_images_dir = "v4_night_" + buffered_images_dir if night_time else \
                                   "v4_day_" + buffered_images_dir
        
        if not os.path.exists(self.buffered_images_dir):
            os.makedirs(self.buffered_images_dir)
        
        self.coco = COCO(ann_file)
        
        max_buffered_image = 7000
        self.buffered_img_ids = []
        count_buffered_image = 0

        print("Start creating image ids...\n")
        self.buffered_img_ids = self.coco.getImgIds()
            
        random.shuffle(self.buffered_img_ids)
        print("Successfully creating image ids!!\n")
        print(f"Number of images: {len(self.buffered_img_ids)}")
        
        self.small_imgs = {1: [], 2: [], 3: [], 4: []}
        self.create_buffered_boxes()
        self.get_small_imgs()
        
        self.buffered_anns = {
            "images": [],
            "annotations": []
        }
    
    def is_small_box(self, ann):
        box = ann["bbox"]
        return (box[2] * box[3] < 56 * 56)

    def get_small_imgs(self):
        for cls in [1, 2, 3, 4]:
            anns = self.coco.loadAnns(self.coco.getAnnIds(catIds = cls))
            for ann in anns:
                if self.is_small_box(ann):
                    self.small_imgs[cls].append(ann)
            print(f"Number of small imgs with class {cls}: {len(self.small_imgs[cls])}")
    
    def transform_img_with_ABR(self, img_id):
        assert img_id in self.buffered_img_ids
        
        img_info = self.coco.loadImgs(ids = img_id)[0]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        gts = self.get_groundtruth(img_id)
        
        is_mixup, is_mosaic = True, False
#         if random.randint(0, 1) == 0:
#             is_mixup = True
#         else:
#             is_mosaic = True
        
#         if is_mixup:
#             img, gts = self.play_mixup(img, gts, self.buffered_boxes[:3])
#         elif is_mosaic:
        boxes = []
        small_id = random.sample([2, 3, 4], 1)[0]
        boxes.extend(random.sample(self.small_imgs[small_id], 1))
        
        for cat_id in range(2, 5):
                boxes.extend(random.sample(self.buffered_boxes[cat_id], 2))
                
        img, gts = self.play_mixup(img, gts, boxes)
        
        return img, gts, is_mixup, is_mosaic

    def create_buffered_boxes(self):        
        self.buffered_boxes = {cls: [] for cls in [1, 2, 3, 4]}

        for cls in [1, 2, 3, 4]:
            all_true_ids = self.coco.getImgIds()
                
            print(f"Class {cls}, Num true ids: {len(all_true_ids)}")
            
            for id in all_true_ids:
                self.buffered_boxes[cls].extend(
                    self.coco.loadAnns(self.coco.getAnnIds(catIds = cls, imgIds = id))
                )

            print(f"Class {cls}, Num base boxes: {len(self.buffered_boxes[cls])}\n")
        print("Get all buffered boxes done!!\n")        
    
    def get_img_bbox(self, img, anns):
        box_imgs, targets = [], [] 
        
        if isinstance(img, np.ndarray):
            im_mean_size = np.mean((img.shape[0], img.shape[1]))
        else:
            im_mean_size = np.mean((img.size[0], img.size[1]))
            
        for ann in anns:
            img_info = self.coco.loadImgs(ids = ann["image_id"])[0]
            img_path = os.path.join(self.images_dir, img_info["file_name"])
            img = np.asarray(Image.open(img_path).convert("RGB")) # (height, width, channel)
            box = ann["bbox"]
            box_img = img[int(box[1]): int(box[1] + box[3]), int(box[0]): int(box[0] + box[2]), :]
            box_img = Image.fromarray(box_img)

            # resize
            box_mean_size = np.mean(box_img.size)
#             box_scale = 1.0

#             box_img = box_img.resize((int(box_scale * box_img.size[0]), int(box_scale * box_img.size[1])))
            target = [0, 0, box_img.size[0], box_img.size[1], ann["category_id"]]

            box_imgs.append(box_img)
            targets.append(np.array([target]))

        return box_imgs, targets
    
    def compute_overlap(self, a, b):
        area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)

        iw = np.minimum(a[2], b[2]) - np.maximum(a[0], b[0]) + 1
        ih = np.minimum(a[3], b[3]) - np.maximum(a[1], b[1]) + 1

        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)

        aa = (a[2] - a[0] + 1)*(a[3] - a[1]+1)
        ba = area

        intersection = iw*ih

        # this parameter can be changes for different datasets
        if intersection/aa > 0.3 or intersection/ba > 0.3:
            return intersection/ba, True

        return intersection/ba, False

    
    def play_mosaic(self, img, anns, bg_size = 0):
        # img: ndarray
        # anns: 4 annotations with coco type

        gt4 = [] # the final groundtruth space
        s_w, s_h = img.size

        yc = int(random.uniform(s_h*0.4, s_h*0.6)) # set the mosaic center position
        xc = int(random.uniform(s_w*0.4, s_w*0.6))

        box_imgs, targets = self.get_img_bbox(img, anns)

        for i, (img, target) in enumerate(zip(box_imgs, targets)):
            (w, h) = img.size
            if i%4==0: # top right
                xc_ = xc+bg_size
                yc_ = yc-bg_size
                img4 = np.full((s_h, s_w, 3), 114., dtype=np.float32)
                x1a, y1a, x2a, y2a = xc_, max(yc_-h, 0), min(xc_+w, s_w), yc_
                x1b, y1b, x2b, y2b = 0, h-(y2a - y1a), min(w, x2a - x1a), h # should corresponding to top left
            elif i%4==1: # bottom left
                xc_ = xc-bg_size
                yc_ = yc+bg_size
                x1a, y1a, x2a, y2a = max(xc_ - w, 0), yc_, xc_, min(s_h, yc_ + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc_, w), min(y2a - y1a, h)
            elif i%4==2: # bottom right
                xc_ = xc+bg_size
                yc_ = yc+bg_size
                x1a, y1a, x2a, y2a = xc_, yc_, min(xc_ + w, s_w), min(s_h, yc_+h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            elif i%4==3: # top left
                xc_ = xc-bg_size
                yc_ = yc-bg_size
                x1a, y1a, x2a, y2a = max(xc_- w, 0), max(yc_ - h, 0), xc_, yc_
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h-(y2a - y1a), w, h

            img4[y1a:y2a, x1a:x2a] = np.asarray(img)[y1b:y2b,x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            gts = np.array(target)
            if len(gts) > 0:
                gts[:, 0] = gts[:, 0] + padw
                gts[:, 1] = gts[:, 1] + padh
                gts[:, 2] = gts[:, 2] + padw
                gts[:, 3] = gts[:, 3] + padh
            gt4.append(gts)

        # Concat/clip gts
        if len(gt4):
            gt4 = np.concatenate(gt4, 0)
            np.clip(gt4[:, 0], 0, s_w, out=gt4[:, 0])
            np.clip(gt4[:, 2], 0, s_w, out=gt4[:, 2])
            np.clip(gt4[:, 1], 0, s_h, out=gt4[:, 1])
            np.clip(gt4[:, 3], 0, s_h, out=gt4[:, 3])

        # Delete too small objects (check again)
        del_index = []

        if len(gt4):
            for col in range(gt4.shape[0]):
                if (gt4[col][2]-gt4[col][0]) <= 2.0 or (gt4[col][3]-gt4[col][1]) <= 2.0:
                    del_index.append(col)
            gt4 = np.delete(gt4, del_index, axis=0)

        curr_image = Image.fromarray(np.uint8(img4)) if len(gt4) else None
        curr_target = gt4 if len(gt4) else None

        return curr_image, curr_target
    
    def play_mixup(self, image, gts, anns, alpha=2.0, beta=5.0):
        """ Mixup the input image

        Args:
            image : PIL.Image - the original image
            gts: np.array - the original box image (xyxy)
            anns : annotations
        Returns:
            mixupped images and targets
        """

        image = np.array(image)
        img_shape = image.shape

        # make sure the image has more than one targets
        # If the only target occupies 75% of the image, we abandon mixupping.
        _MIXUP=True
        if gts.shape[0] == 1:
            img_w = gts[0][2]-gts[0][0]
            img_h = gts[0][3]-gts[0][1]
            if (img_shape[1]-img_w)<(img_shape[1]*0.25) and (img_shape[0]-img_h)<(img_shape[0]*0.25):
                _MIXUP=False

        if _MIXUP: # 
            Lambda = torch.distributions.beta.Beta(alpha, beta).sample().item()
            num_mixup = 8 # more mixup boxes but not all used

            mixup_count = 0
            box_imgs, targets = self.get_img_bbox(image, anns)
            for c_img, c_gt in zip(box_imgs, targets):
                c_img = np.asarray(c_img)
                _c_gt = c_gt.copy()

                # assign a random location
                pos_x = random.randint(0, int(img_shape[1] * 0.6))
                pos_y = random.randint(0, int(img_shape[0] * 0.4))
                new_gt = [c_gt[0][0] + pos_x, c_gt[0][1] + pos_y, c_gt[0][2] + pos_x, c_gt[0][3] + pos_y]

                restart = True
                overlap = False
                max_iter = 0
                # compute the overlap with each gts in image
                while restart:
                    for g in gts:      
                        _, overlap = self.compute_overlap(g, new_gt)
                        if max_iter >= 20:
                            # if iteration > 20, delete current choosed sample
                            restart = False
                        elif max_iter < 10 and overlap:
                            pos_x = random.randint(0, int(img_shape[1] * 0.6))
                            pos_y = random.randint(0, int(img_shape[0] * 0.4))
                            new_gt = [c_gt[0][0] + pos_x, c_gt[0][1] + pos_y, c_gt[0][2] + pos_x, c_gt[0][3] + pos_y]
                            max_iter += 1
                            restart = True
                            break
                        elif 20 > max_iter >= 10 and overlap:
                            # if overlap is True, then change the position at right bottom
                            pos_x = random.randint(int(img_shape[1] * 0.4), img_shape[1])
                            pos_y = random.randint(int(img_shape[0] * 0.6), img_shape[0])
                            new_gt = [pos_x-(c_gt[0][2]-c_gt[0][0]), pos_y-(c_gt[0][3]-c_gt[0][1]), pos_x, pos_y]
                            max_iter += 1
                            restart = True
                            break
                        else:
                            restart = False

                if max_iter < 20:
                    a, b, c, d = 0, 0, 0, 0
                    if new_gt[3] >= img_shape[0]:
                        # at bottom right new gt_y is or not bigger
                        a = new_gt[3] - img_shape[0]
                        new_gt[3] = img_shape[0]
                    if new_gt[2] >= img_shape[1]:
                        # at bottom right new gt_x is or not bigger
                        b = new_gt[2] - img_shape[1]
                        new_gt[2] = img_shape[1]
                    if new_gt[0] < 0:
                        # at top left new gt_x is or not bigger
                        c = -new_gt[0]
                        new_gt[0] = 0
                    if new_gt[1] < 0:
                        # at top left new gt_y is or not bigger
                        d = -new_gt[1]
                        new_gt[1] = 0

                     # Use the formula by the paper to weight each image
                    if a == 0 and b == 0:
                        if c == 0 and d == 0:
                            image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = c_img[:, :]
                        elif c != 0 and d == 0:
                            image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = c_img[:, c:]
                        elif c == 0 and d != 0:
                            image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = c_img[d:, :]
                        else:
                            image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = c_img[d:, c:]

                    elif a == 0 and b != 0:
                        image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = c_img[:, :-b]
                    elif a != 0 and b == 0:
                        image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = c_img[:-a, :]
                    else:
                        image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = c_img[:-a, :-b]

                    _c_gt[0][:-1] = new_gt
                    if gts.shape[0] == 0:
                        gts = _c_gt
                    else:
                        gts = np.insert(gts, 0, values=_c_gt, axis=0)

                mixup_count += 1
                if mixup_count>=6:
                    break

        curr_image = Image.fromarray(np.uint8(image))
        curr_target = gts.astype(int)

        return curr_image, curr_target
    
    def save_buffer_image_and_annotations(self):
        start_ann_id = 0
        start_img_id = 0

        print("Start creating image with boxes...\n")
        first = "v4_night_" if self.night_time else "v4_day_"
        
        for img_id in tqdm(self.buffered_img_ids, total = len(self.buffered_img_ids)):
            try:
                img, gts, is_mixup, is_mosaic = self.transform_img_with_ABR(img_id)
            except:
                continue
            
            buffered_im_name = (first + "image_{}.jpg").format(start_img_id)
            self.buffered_anns["images"].append({
                "id": start_img_id,
                "file_name": buffered_im_name,
                "height": img.size[1],
                "width": img.size[0]
            })   
            
            for gt in gts:
                ann = {}
                ann["id"] = start_ann_id
                ann["image_id"] = start_img_id
                ann["category_id"] = gt[4]
                bbox = gt[:4]; bbox[2] -= bbox[0]; bbox[3] -= bbox[1]
                ann["bbox"] = list(bbox)
                ann["area"] = bbox[2] * bbox[3]
                ann["iscrowd"] = 0
                ann["bbox_mode"] = 0
                ann["segmentation"] = []
                
                self.buffered_anns["annotations"].append(ann)
                start_ann_id += 1

            # save the box image
            img.save(os.path.join(self.buffered_images_dir, buffered_im_name))
            start_img_id += 1
        
        orig_anns = {
            "images": [],
            "annotations": [],
            "categories": [
                 {'id': 1, 'name': 'motor'},
                 {'id': 2, 'name': 'car'},
                 {'id': 3, 'name': 'bus'},
                 {'id': 4, 'name': 'container'}
            ]
        }
        
        orig_anns["images"].extend(self.buffered_anns["images"])
        orig_anns["annotations"].extend(self.buffered_anns["annotations"])
        
        buffer_anns = json.dumps(orig_anns, cls = NpEncoder)
        json_file = first + "buffer.json"
        with open(json_file, "w") as file:
            file.write(buffer_anns)
        
    def get_groundtruth(self, img_id):
        img_info = self.coco.loadImgs(ids = img_id)[0]
        annots = self.coco.loadAnns(ids = self.coco.getAnnIds(imgIds = img_id))
        gts = []
        for annot in annots:
            box = annot["bbox"]
            gts.append([box[0], box[1], box[0] + box[2], box[1] + box[3], annot["category_id"]])
        
        return np.asarray(gts)
    
    def visualize(self, img_id):
        img_info = self.coco.loadImgs(ids = img_id)[0]
        
        img, target, is_mixup, is_mosaic = self.transform_img_with_ABR(img_id)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        for box in target:
            img = cv2.rectangle(img, \
                                (int(box[0]), int(box[1])), \
                                (int(box[2]), int(box[3])),\
                                color = (255, 0, 0),
                                thickness = 10)
        plt.imshow(img)
