"""
Add utility of caching images on memory
"""

from torchvision.datasets import CocoDetection as CCD
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO
from termcolor import colored, cprint
import random
import numpy as np
import torch

class CocoCache(CCD):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root,
        annFile,
        transform=None,
        target_transform=None,
        transforms=None,
        cache_mode=False,
        ids_list=None,
        class_ids=None,
        buffer_ids=None,
        buffer_rate=None,
        buffer_mode=None,
    ):
        super(CocoCache, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)

        self.ids = list(sorted(self.coco.imgs.keys())) if ids_list == None else ids_list

        if not isinstance(class_ids, list):
            class_ids = list(class_ids)
        self.class_ids = class_ids

        if class_ids is not None and ids_list == None:
            self.ids = []

            for c_idx in self.class_ids:
                img_ids = self.coco.getImgIds(catIds=c_idx)
                self.ids.extend(img_ids)

            cprint(
                f"Original Images: {len(set(self.ids))}",
                "green",
                "on_red",
            )

            if buffer_mode:
                total_buffer = 0
                self.buffer_ids = buffer_ids

                for b_idx in self.buffer_ids:
                    buffer_ids = self.coco.getImgIds(catIds=b_idx)
                    buffer_size = int(len(buffer_ids) * buffer_rate)
                    total_buffer += buffer_size
                    self.ids.extend(buffer_ids[:buffer_size])
                cprint(
                    f"Buffer Images: {total_buffer}\n{len(self.buffer_ids)} Buffer Classes: {self.buffer_ids}",
                    "green",
                    "on_red",
                )
                print("---------------------------------")

            self.ids = list(set(self.ids))

        self.cache_mode = cache_mode

        if cache_mode:
            self.cache = {}
            self.cache_images()
            self.curr_cls = class_ids
            self.start_ann_id = int(1e6)
            self.images_dir = root

            self.small_imgs = {}
            num_small_imgs = 0
            for cls in [1, 2, 3, 4]:
                num_small_imgs += len(self.small_imgs[cls])

            self.get_small_imgs()
            
            cprint(
                f"Number small images: {num_small_imgs}",
                "red",
                "on_cyan",
            )

        cprint(
            f"Total Images: {len(self.ids)}\n{len(self.class_ids)} Task Classes: {self.class_ids}",
            "red",
            "on_cyan",
        )

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            path = self.coco.loadImgs(img_id)[0]["file_name"]

            with open(os.path.join(self.root, path), "rb") as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), "rb") as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert("RGB")

        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        # TODO : 기존에는 모든 존재하는 이미지 ID를 가져와서 사용하던 것을 변환 -> 미리 선언한 Class에 해당하는 이미지들만 선별

        if (
            self.class_ids is not None
        ):  # 클래스 IDS에 해당하는 Target만 가져오는 것이 핵심
            target = [
                value
                for value in coco.loadAnns(coco.getAnnIds(img_id))
                if value["category_id"] in self.class_ids
            ]
        else:
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(int(img_id))[0]["file_name"]
        img = self.get_image(path)

        if random.randint(0, 1) == 0:
            img, target = self.get_new_image(img, target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
    
    def get_new_image(self, img, anns):
        samples = random.sample([1, 2, 3, 4], 2)
        cls_1, cls_2 = samples
        rand_box_1 = random.sample(self.small_imgs[cls_1], 1)
        rand_box_2 = random.sample(self.small_imgs[cls_2], 1)
        
        small_anns = rand_box_1 + rand_box_2
        new_img, new_anns = self.copy_paste(img, anns, small_anns)
        
        return new_img, new_anns
    
    def is_small_box(self, ann):
        box = ann["bbox"]
        return (box[2] * box[3] < 64 * 64)

    def get_small_imgs(self):
        for cls in [1, 2, 3, 4]:
            anns = self.coco.loadAnns(self.coco.getAnnIds(catIds = cls))
            for ann in anns:
                if self.is_small_box(ann):
                    self.small_imgs[cls].append(ann)
    
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
    
    def get_img_bbox(self, img, anns):
        box_imgs, targets = [], [] 
        
        if isinstance(img, np.ndarray):
            im_mean_size = np.mean((img.shape[0], img.shape[1]))
        else:
            im_mean_size = np.mean((img.size[0], img.size[1]))
            
        for ann in anns:
            img_info = self.coco.loadImgs(ids = ann["image_id"])[0]
            img = np.asarray(self.get_image(img_info["file_name"])) # (height, width, channel)
            box = ann["bbox"]
            box_img = img[int(box[1]): int(box[1] + box[3]), int(box[0]): int(box[0] + box[2]), :]
            box_img = Image.fromarray(box_img)
            target = [0, 0, box_img.size[0], box_img.size[1], ann["category_id"]]

            box_imgs.append(box_img)
            targets.append(np.array([target]))

        return box_imgs, targets
        
    def get_groundtruth(self, anns):
        gts = []
        for annot in anns:
            box = annot["bbox"]
            gts.append([box[0], box[1], box[0] + box[2], box[1] + box[3], annot["category_id"]])
        
        return np.asarray(gts)
            
    def copy_paste(self, image, curr_anns, anns, alpha=2.0, beta=5.0):
        image = np.array(image)
        gts = self.get_groundtruth(curr_anns)
        img_shape = image.shape
        _MIXUP = True

        if _MIXUP: # 
            Lambda = torch.distributions.beta.Beta(alpha, beta).sample().item()
            num_mixup = 2 # more mixup boxes but not all used

            mixup_count = 0
            box_imgs, targets = self.get_img_bbox(image, anns)
            for c_img, c_gt in zip(box_imgs, targets):
                c_img = np.asarray(c_img)
                _c_gt = c_gt.copy()

                # assign a random location
                pos_x = random.randint(0, int(img_shape[1] * 0.7))
                pos_y = random.randint(0, int(img_shape[0] * 0.7))
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
                            pos_x = random.randint(0, int(img_shape[1] * 0.7))
                            pos_y = random.randint(0, int(img_shape[0] * 0.7))
                            new_gt = [c_gt[0][0] + pos_x, c_gt[0][1] + pos_y, c_gt[0][2] + pos_x, c_gt[0][3] + pos_y]
                            max_iter += 1
                            restart = True
                            break
                        elif 20 > max_iter >= 10 and overlap:
                            # if overlap is True, then change the position at right bottom
                            pos_x = random.randint(int(img_shape[1] * 0.7), img_shape[1])
                            pos_y = random.randint(int(img_shape[0] * 0.7), img_shape[0])
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

                    # Combine the images
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
                    gts = _c_gt if gts.shape[0] == 0 else np.insert(gts, 0, values=_c_gt, axis=0)
        
                mixup_count += 1
                if mixup_count>=2:
                    break

        curr_image = Image.fromarray(np.uint8(image))
        new_anns = []
        img_id = curr_anns[0]["image_id"]
        for gt in gts.astype(int):
            ann = {}
            ann["id"] = self.start_ann_id
            ann["image_id"] = img_id
            ann["category_id"] = gt[4]
            bbox = gt[:4]; bbox[2] -= bbox[0]; bbox[3] -= bbox[1]
            ann["bbox"] = list(bbox)
            ann["area"] = bbox[2] * bbox[3]
            ann["iscrowd"] = 0
            ann["bbox_mode"] = 0
            ann["segmentation"] = []
            new_anns.append(ann)
            self.start_ann_id += 1

        return curr_image, new_anns
