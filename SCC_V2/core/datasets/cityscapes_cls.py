import os
import json
from typing import Optional, List, Dict, Any

import torch
from torch.utils import data
import numpy as np
from PIL import Image
import cv2
from pycocotools import mask as mask_utils
from collections import Counter

class CityscapesCLSDataSet(data.Dataset):
    def __init__(
        self,
        data_root: str,
        data_list: str,                   
        max_iters: Optional[int] = None,
        num_classes: int = 19,
        split: str = "train",
        transform=None,                   
        ignore_label: int = 255,
        cfg=None,
        debug: bool = False,
        resample_alpha: float = 0.7  #  
    ):
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.transform = transform
        self.ignore_label = ignore_label
        self.debug = debug

        with open(data_list, "r") as f:
            global_by_cid: Dict[str, List[Dict[str, Any]]] = json.load(f)

        self.entries: List[Dict[str, Any]] = []
        for cid_str, items in global_by_cid.items():
            cid_int = int(cid_str)
            for e in items:
                rec = dict(e)  
                rec["cid"] = cid_int
                self.entries.append(rec)

        if len(self.entries) == 0:
            raise ValueError("No entries loaded from JSON. Check data_list.")

        self.resample_alpha = float(resample_alpha)
        self.cid_counts = Counter([e["cid"] for e in self.entries])
        self.sample_weights = [
            (1.0 / self.cid_counts[e["cid"]]) ** self.resample_alpha
            for e in self.entries
        ]
        

    def __len__(self):
        return len(self.entries)

    def _resolve_path(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(self.data_root, p)

    def __getitem__(self, index: int):
        if self.debug:
            index = 0

        e = self.entries[index]

        crop_path = self._resolve_path(e["crop_path"])
        image = Image.open(crop_path).convert("RGB")
        w_img, h_img = image.size  # (W,W)

        rle = e["mask"]
        mask_np = mask_utils.decode(rle).astype(np.uint8)  # [H, W] 0/1

        if mask_np.shape[1] != w_img or mask_np.shape[0] != h_img:
            mask_np = cv2.resize(mask_np, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        mask_pil = Image.fromarray(mask_np * 255).convert("L")  # 0/255

        if self.transform is not None:
            image, mask, _ = self.transform(image, mask_pil)

        name = f'{e.get("image_id", "unknown")}::{os.path.basename(crop_path)}'

        sam_psd_labels = torch.tensor([int(e.get("psd_label", e["cid"]))], dtype=torch.long)
        sam_gt_labels  = torch.tensor([int(e.get("gt_label", 255))], dtype=torch.long)

        bbox_xywh = e["bbox_xywh"]  # [x, y, w, h]
        x1 = bbox_xywh[0]
        y1 = bbox_xywh[1]
        x2 = x1 + bbox_xywh[2]
        y2 = y1 + bbox_xywh[3]
    
        # 转换成 tensor 格式：[x1, y1, x2, y2]
        bbox_tensor = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        bbox_tensor = torch.tensor(bbox_xywh, dtype=torch.float32)
        
        return image, mask, sam_psd_labels, sam_gt_labels, name, bbox_tensor
