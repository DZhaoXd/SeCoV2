import os
import numpy as np
import cv2

from tqdm import tqdm
from skimage.measure import label as sklabel
from utils import mkdir, unclip, cal_four_para_bbox, get_mini_boxes, Calculate_purity, get_color_pallete, save_image_with_anns
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import argparse
from collections import Counter
from pycocotools import mask as mask_utils
import json
from collections import defaultdict

# python seco_sam.py --id-list-path  './splits/cityscapes/HRDA_seco/labeled.txt'
# --data
#   --psd label
#   --leftImg8bit

# save to HRDA_seco/

parser = argparse.ArgumentParser(description='Semantic Connectivity-Driven Pseudo-labeling for Cross-domain Segmentation.')
parser.add_argument('--id-list-path', type=str, required=True)
parser.add_argument('--class-num', type=int, default=19)
parser.add_argument('--image-root-path', type=str, default='/data/zd/SeCoV2/data/UDA/cityscapes/train/')
parser.add_argument('--gt-root-path', type=str, default='/data/zd/SeCoV2/data/UDA/cityscapes/train_gt/')
parser.add_argument('--psd-root-path', type=str, default='/data/zd/SeCoV2/data/UDA/G2C/DTST_PL/CS_no_thr/train/')
# parser.add_argument('--sam-checkpoint', type=str, default="/data/zd/SAM/sam_vit_h_4b8939.pth")
# parser.add_argument('--sam-type', type=str, default="vit_h")
parser.add_argument('--sam-checkpoint', type=str, default="/data/zd/SAM/sam_vit_b_01ec64.pth")
parser.add_argument('--sam-type', type=str, default="vit_b")

def main():
    args = parser.parse_args()
    # 全局容器，key 为字符串 cid（JSON 兼容），value 为列表
    global_by_cid = defaultdict(list)

    ## init sam
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint)
    sam.to(device="cuda")
    predictor = SamPredictor(sam)
    SAM_GRIDS = False
    if SAM_GRIDS:
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            box_nms_thresh = 0.5,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )
    else:
        mask_generator = SamAutomaticMaskGenerator(model=sam,
                                                   box_nms_thresh = 0.5,
                                                   min_mask_region_area=500,)
    
    with open(args.id_list_path, 'r') as f:
        list_ids = f.read().splitlines()

    ### STUFF   
    stuff_region_total = Counter()
    stuff_region_correct = Counter()
    ### THING
    thing_region_total = Counter()
    thing_region_correct = Counter()
        
    all_sam_mask_infos = []
    save_root_path = (args.psd_root_path.rstrip('/') + '_sam_cls') if args.psd_root_path.endswith('/') else args.psd_root_path

    for id in tqdm(list_ids):
        ### read data
        image = Image.open(os.path.join(args.image_root_path, id)).convert('RGB')
        psd_label =  Image.open(os.path.join(args.psd_root_path, id))
        gt =  Image.open(os.path.join(args.gt_root_path, id.replace('_leftImg8bit','_gtFine_labelTrainIds')))
        
        target_size = (1024, 512)
        image = np.array(image.resize(target_size))
        gt = np.array(gt.resize(target_size))
        psd_label = np.array(psd_label.resize(target_size))
        
        ### init label sets
        if args.class_num == 19:
            STUFF_LABEL = [0,1,2,3,4,8,9,10,11,12,13,14,15,16,17,18,5,6,7]  # semantic alignment
            THING_LABEL = [5,6,7]  # box + point prompt
        elif args.class_num == 16:
            STUFF_LABEL = [0,1,2,3,4,8,9]  # semantic alignment
            THING_LABEL = [5,6,7,10,11,12,13,14,15]  # box + point prompt            
            
        unique_stuff_list = []
        unique_thing_list = []
        for cid in np.unique(psd_label):
            if cid in STUFF_LABEL:
                unique_stuff_list.append(cid)
            if cid in THING_LABEL:
                unique_thing_list.append(cid)
                
        ### init return vars
        all_point = []
        all_label = []
        all_boxes = []
        filled_mask = np.zeros_like(psd_label)
        gt_final_mask = np.ones_like(psd_label) * 255
        final_psd_label = np.copy(psd_label)  # Initialize final_psd_label with psd_label

        ### STUFF  ---->  semantic alignment
        predictor.set_image(image)
        sam_masks = mask_generator.generate(image)
        stuff_align_thr = 0.1
        for mask_ in sam_masks:
            mask = mask_['segmentation']
            Proportion = []
            for cid in unique_stuff_list: 
                Proportion.append(np.mean(psd_label[(mask>0) * (psd_label!=255)] == cid))       
            try:
                max_idx, max_pro = np.argmax(np.array(Proportion)), np.max(np.array(Proportion))
                if max_pro > stuff_align_thr:
                    class_id = unique_stuff_list[max_idx]
                    final_psd_label[mask > 0] = class_id  # Update final_psd_label with SAM mask
                    filled_mask[mask > 0] += 1
                    
                    region_mask = (mask > 0)
                    gt_region = gt[region_mask]
                    gt_region_valid = gt_region[gt_region != 255]
                    if len(gt_region_valid) > 0:
                        majority_class = np.bincount(gt_region_valid).argmax()
                        gt_final_mask[region_mask] = majority_class

            except Exception as e:
                continue
                
        ###  THING_LABEL_SMALL (instance )  ---->  box + point prompt
        minum_psd_nums = 2
        purity = Calculate_purity(psd_label, class_num=args.class_num)
        for _, label_id in enumerate(unique_thing_list):
            mask = psd_label == label_id
            
            if np.sum(mask) > minum_psd_nums:
                masknp = mask.astype(int) 
                seg, forenum = sklabel(masknp, background=0, return_num=True, connectivity=2)
                for i in range(forenum):
                    instance_id = i + 1
                    if np.sum(seg == instance_id) < minum_psd_nums:
                        continue
                    ins_mask = seg == instance_id
                    cont, _ = cv2.findContours(ins_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    cont = sorted(cont, key=lambda c: cv2.contourArea(c), reverse=True)

                    points, _ = get_mini_boxes(cont[0])
                    points = np.array(points)
                    box = unclip(points, unclip_ratio=1.25)
                    if len(box) == 0: continue
                    x, y, w, h = cal_four_para_bbox(ins_mask, box.reshape(-1, 2))
                    input_box = np.array([x, y,  x+w, y+h])
                    all_boxes.append([x, y,  x+w, y+h])
        
                    label_where = np.where(seg == instance_id)
                    top_points_nums = 2
                    select_point, select_lbl = [], []
                    point_, score_, = [], []
                    divided = len(label_where[1]) // top_points_nums
                    cur_points_nums = 1
                    for idx, xy in enumerate(zip(label_where[1], label_where[0])):
                        point_.append(xy)
                        score_.append(purity[xy[1], xy[0]])
                        if idx == divided * cur_points_nums:
                            min_indices_h = np.argmin(np.array(score_))
                            
                            select_point.append(point_[min_indices_h])
                            select_lbl.append(1)
                            
                            cur_points_nums += 1
                            score_ = []
                            point_ = []
                            
                    if len(select_point) > 0:
                        masks, score, _ = predictor.predict(
                            point_coords= np.array(select_point),
                            point_labels= np.array(select_lbl),
                            box=input_box,
                            multimask_output=False,
                        )
                    
                        all_point += select_point
                        all_label += [x * label_id for x in select_lbl]
                        
                        final_psd_label[masks[0]] = label_id  # Update final_psd_label
                        filled_mask[masks[0]] += 1 
                        
                        
                        ##  
                        gt_region = gt[ins_mask]
                        gt_region_valid = gt_region[gt_region != 255]
                        if len(gt_region_valid) > 0:
                            majority_class = np.bincount(gt_region_valid).argmax()
                            gt_final_mask[ins_mask] = majority_class
                                
        # Run connected components analysis for final_psd_label
        for cid in np.unique(final_psd_label):
            if cid != 255:  # Ignore background
                region_mask = final_psd_label == cid
                
                # Check if the region has more than 1000 pixels
                if np.sum(region_mask) <= 1000:
                    continue  # Skip if region has less than or equal to 1000 pixels
                
                # Use skimage's label function to detect connected components
                labeled_mask = sklabel(region_mask, connectivity=2, background=0)
                
                # Get the number of connected components by checking the max label index
                num_components = labeled_mask.max()  # The max label value corresponds to the number of components
                
                for component_id in range(1, num_components + 1):
                    component_mask = labeled_mask == component_id
                    
                    # Filter out components that are smaller than 50 pixels
                    if np.sum(component_mask) <= 100:
                        continue  # Skip if component has less than or equal to 50 pixels
                    
                    # Find the bounding box for the component_mask
                    x, y, w, h = cv2.boundingRect(component_mask.astype(np.uint8))

                    # —— 精确外扩到至少 64x32（边界安全）——
                    min_w, min_h = 64, 32
                    if w < min_w or h < min_h:
                        pad_w = max(0, (min_w - w) // 2)
                        pad_h = max(0, (min_h - h) // 2)
                
                        x0 = max(0, x - pad_w)
                        y0 = max(0, y - pad_h)
                        x1 = min(image.shape[1], x + w + pad_w)
                        y1 = min(image.shape[0], y + h + pad_h)
                
                        x, y = x0, y0
                        w, h = x1 - x0, y1 - y0  # 可能比目标略小/大，但已尽可能扩到 >=64x32（受边界限制）
                    # —— 结束外扩 —— 

                    
                    # Crop the component_mask using the bounding box
                    cropped_component_mask = component_mask[y:y+h, x:x+w]
                    cropped_image = image[y:y+h, x:x+w]
                    
                    # Resize the cropped mask to 128x64
                    target_size = (128, 64)  # (W, H)
                    resized_cropped_mask = cv2.resize(cropped_component_mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
                    resized_image = cv2.resize(cropped_image, target_size)
                    
                    
                    # Encode the cropped component mask (bounding box mask)
                    encoded_mask = mask_utils.encode(np.asfortranarray(resized_cropped_mask))
                    encoded_mask['counts'] = encoded_mask['counts'].decode('utf-8')  # Decode to make it JSON serializable
                    
                    # Save the cropped and resized mask information into sam_mask_infos
                    
                    new_image_name = f"cid_{cid}_cptid_{component_id}_image_{id.split('/')[-1]}".replace('png', 'jpg')
                    new_ins_name = os.path.join(save_root_path, 'ins_mask', new_image_name)
                    mkdir(os.path.dirname(new_ins_name))
                    Image.fromarray(resized_cropped_mask * 255).save(new_ins_name)  # Save the mask image
                    save_image_path = os.path.join(save_root_path, 'image', new_image_name)
                    mkdir(os.path.dirname(save_image_path))
                    Image.fromarray(resized_image).save(save_image_path)
                    
                    # Update the region total and correct counts
                    stuff_region_total[cid] += 1  # Total count +1
        
                    # For GT comparison
                    majority_class = 255
                    gt_region = gt[component_mask]
                    gt_region_valid = gt_region[gt_region != 255]
                    if len(gt_region_valid) > 0:
                        majority_class = np.bincount(gt_region_valid).argmax()
                        gt_final_mask[component_mask] = majority_class
                        if majority_class == cid:
                            stuff_region_correct[cid] += 1  # Correct count +1
                    
                    cid_key = str(int(cid))  # 确保 JSON 兼容
                    
                    entry = {
                        "image_id": id,                           # 当前图片ID
                        "bbox_xywh": [int(x), int(y), int(w), int(h)],   # 原图坐标系下的裁剪框
                        "mask": encoded_mask,                     # 裁剪+resize 后的RLE
                        "crop_path": save_image_path,                   # 裁剪后图片路径
                        "crop_size": [cropped_image.shape[1], cropped_image.shape[0]],  # [W,H]
                        "resize_size": [128, 64],                  # 最终统一尺寸
                        "psd_label": int(cid),
                        "gt_label": int(majority_class)
                    }
                    
                    # 加入全局字典
                    global_by_cid[cid_key].append(entry)


                    
        ### gt_final_mask  show
        # final_mask_color = get_color_pallete(final_psd_label, "city")
        # gt_final_mask_color = get_color_pallete(gt_final_mask, "city")
        # ori_psd_label = get_color_pallete(psd_label, "city")
        # layers = [
        #     Image.fromarray(image),  # 原始图像
        #     ori_psd_label,  # 原始伪标签
        #     final_mask_color,  # 最终的掩膜颜色
        #     gt_final_mask_color  # Ground truth 掩膜颜色
        # ]
        # combined_image = Image.new('RGB', (final_mask_color.width * len(layers), final_mask_color.height))
        # for i, layer in enumerate(layers):
        #     combined_image.paste(layer, (final_mask_color.width * i, 0))
        # gt_save_path = os.path.join(save_root_path + "_gt", id.replace('png', 'jpg'))   
        # mkdir(os.path.dirname(gt_save_path))
        # combined_image.save(gt_save_path)
        
        # 转普通 dict 以便 JSON 存储
        global_by_cid_json = dict(global_by_cid)
        
        json_save_path = save_root_path + "_mask_psa_infos_cls.json"
        with open(json_save_path, "w") as f:
            json.dump(global_by_cid_json, f)

        #### print some infos
        stuff_total = sum(stuff_region_total.values())
        stuff_correct = sum(stuff_region_correct.values())
        thing_total = sum(thing_region_total.values())
        thing_correct = sum(thing_region_correct.values())
        
        stuff_acc = stuff_correct / (stuff_total + 1e-6)
        thing_acc = thing_correct / (thing_total + 1e-6)
        overall_acc = (stuff_correct + thing_correct) / (stuff_total + thing_total + 1e-6)
        
        print(f"[{id}] Stuff ACC: {stuff_acc:.3f} ({stuff_correct}/{stuff_total}), "
              f"Thing ACC: {thing_acc:.3f} ({thing_correct}/{thing_total}), "
              f"Overall ACC: {overall_acc:.3f}")

    #### print total infos
    print(f"[{id}] STUFF REGION ACCURACY PER CLASS:")
    for cid in stuff_region_total:
        total = stuff_region_total[cid]
        correct = stuff_region_correct[cid]
        acc = correct / total if total > 0 else 0.0
        print(f"  STUFF Class {cid}: {acc:.3f} ({correct}/{total})")
    
    print(f"[{id}] THING REGION ACCURACY PER CLASS:")
    for cid in thing_region_total:
        total = thing_region_total[cid]
        correct = thing_region_correct[cid]
        acc = correct / total if total > 0 else 0.0
        print(f"  THING Class {cid}: {acc:.3f} ({correct}/{total})")

if __name__ == '__main__':
    main()