import argparse
import os
import datetime
import logging
import time
import math
import numpy as np
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import re
import cv2

from PIL import Image
import torch
import torch.nn.functional as F

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_model, build_adversarial_discriminator, build_feature_extractor, build_classifier, build_GraphVAE, GraphBuilder
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU, get_color_pallete
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from core.utils.sam import unclip, cal_four_para_bbox, get_mini_boxes, Calculate_purity, text_save
from skimage.measure import label as sklabel
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

def _to_py_name(name_i):
    # name_i can be str / bytes / 0-dim tensor / tensor of char codes, etc.
    if isinstance(name_i, str):
        return name_i
    if isinstance(name_i, bytes):
        return name_i.decode('utf-8', errors='ignore')
    if torch.is_tensor(name_i):
        name_i = name_i.detach().cpu()
        # 0-d numerical tensor
        if name_i.dim() == 0:
            return str(name_i.item())
        # Array of character codes (uint8) -> string
        if name_i.dtype == torch.uint8:
            try:
                return bytes(name_i.tolist()).decode('utf-8', errors='ignore')
            except Exception:
                return str(name_i.tolist())
        # Convert all other cases to string
        return str(name_i.tolist())
    # Fallback
    return str(name_i)



def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def inference(feature_extractor, classifier, image, label, flip=True, num_classes=19):
    size = label.shape[-2:]  # Get the last two dimensions of label
    if flip:
        image = torch.cat([image, torch.flip(image, [3])], 0)  # Apply flipping

    with torch.no_grad():
        # Pass through feature extractor and classifier
        output = classifier(feature_extractor(image)[1], size)
    
    # Upsample the output to match the label size
    output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
    
    # Compute softmax to get the probability for each class
    output = F.softmax(output, dim=1)  # Softmax over all classes for each pixel
    
    if flip:
        # If flip is enabled, average forward and flipped outputs, then restore dimensions
        output = (output[0] + output[1].flip(2)) / 2
        output = output.unsqueeze(0)  # Add batch_size dimension
    else:
        output = output[0].unsqueeze(0)  # Without flip, take forward output and add batch_size dimension

    return output  # Returns output of shape [batch_size, num_classes, w, h]





@torch.no_grad()
def filter_correct_and_assemble(
    cfg, saveres, 
    mode='inference',
    keep_ratio_per_class: float = 0.8,
    correct_conf_thres: float = 0.95,
    base_size: Tuple[int, int] = (1024, 512),   # (W, H)
    use_history: bool = False,
    history_bank: Tuple[torch.Tensor, torch.Tensor] = None,  # (mem_feats[n,K,C], mem_coords[n,K,4])
    unc_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),  # (w_rec, w_ent, w_kl) 仅用于 compute_uncertainty 备选
    tau: float = 2.0,
    w_min: float = 0.1,
    ignore_index: int = 255,
):
    """
    Two stages:
      1) Iterate over the dataset: record (name, coord, noisy_label, pred_label, conf, unc) for each sample
      2) Within-class selection of top keep_ratio_per_class + high-confidence correction; iterate again and only fill back the "kept samples"

    Returns:
      canvases: dict[name] = {'label': LongTensor[H,W], 'conf': FloatTensor[H,W]}
      stats:    dict with summary statistics
      decisions: list[dict] with final decision for each sample (kept or not, corrected label, etc.)
    """

    # 解包模型
    print('test mode', mode)
    logger = logging.getLogger("SCC_Unc.tester")
    logger.info("Start testing")
    device = torch.device(cfg.MODEL.DEVICE)

    feature_extractor = build_feature_extractor(cfg)
    feature_extractor.to(device)

    classifier = build_classifier(cfg)
    classifier.to(device)
    
    gb = GraphBuilder()  
    graph_vae = build_GraphVAE(cfg)
    graph_vae.to(device)
    
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(feature_extractor_weights)
        classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)

    feature_extractor.eval()
    classifier.eval()
    graph_vae.eval()
    
    torch.cuda.empty_cache()  # TODO check if it helps
    dataset_name = cfg.DATASETS.TEST
    output_folder = '.'
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)

    test_data = build_dataset(cfg, mode='test', is_source=False)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, sampler=None)


    # Historical bank (not used here according to your forward snippet; kept for compatibility/extension)
    mem_feats, mem_coords = (None, None)
    if use_history and history_bank is not None:
        mem_feats, mem_coords = history_bank

    W0, H0 = base_size

    # --------------------------
    # Stage 1: Iterate over dataset and collect meta-information
    # --------------------------

    entries: List[Dict] = []   
    total = 0

    for batch in tqdm(test_loader, desc="Testing", ncols=100):
        # 期望 batch: x, sam_masks, sam_psd_labels, sam_gt_labels, coordinates, names
        x, sam_masks, sam_psd_labels, sam_gt_labels, names, coordinates = batch
        x = x.to(device, non_blocking=True)
        coordinates = coordinates.to(device, non_blocking=True).float()          # [B,4] 绝对像素 (x,y,w,h)

        # 统一 sam_masks -> [B,1,Hm,Wm]
        if sam_masks.dim() == 3:
            sam_masks = sam_masks.unsqueeze(1)
        sam_masks = sam_masks.to(device, non_blocking=True).float()              # [B,1,Hm,Wm]

        sam_psd_labels = sam_psd_labels.to(device, non_blocking=True).long()     # [B]
        
        with torch.no_grad():
            features = feature_extractor(x, sam_masks)
            _, sam_feat = classifier(features, sam_masks)
            B = sam_feat.size(0)
            adj, node_feats, _ = gb.build_graph(
                features=sam_feat, coordinates=coordinates, history_features=None, history_coords=None
            )
            recon_x, ori_x, mu, logvar, logits = graph_vae(adj, sam_feat.unsqueeze(0), labels=None)
            U, _, _ = graph_vae.compute_uncertainty(
                recon_x, ori_x, logits, mu=mu, logvar=logvar,
                w_rec=1.0, w_ent=1.0, w_kl=1.0, tau=2.0, w_min=0.1, ignore_index=255
            )
        
        if U.dim() == 2 and U.size(0) == 1:
            U = U.squeeze(0)  # [B]
        elif U.dim() == 1:
            pass  # 已是 [B]
        else:
            raise RuntimeError(f"Unexpected U shape: {tuple(U.shape)}")

        # Prediction and confidence (computed from logits directly)
        probs = torch.softmax(logits, dim=-1)    # [1,B,K]
        conf, pred = probs.max(dim=-1)           # [1,B]
        conf = conf.squeeze(0)                   # [B]
        pred = pred.squeeze(0).to(torch.long)    # [B]

        # 5) meta info
        for i in range(B):
            name_py = _to_py_name(names[i] if isinstance(names, (list, tuple)) else names[i]).split('::')[0]
            coord_py = tuple(float(v) for v in coordinates[i].detach().cpu().tolist())  # (x,y,w,h) 全是 float/py 标量
            noisy_py = int(sam_psd_labels[i].detach().cpu().item())
            pred_py  = int(pred[i].detach().cpu().item())
            conf_py  = float(conf[i].detach().cpu().item())
            unc_py   = float(U[i].detach().cpu().item())
        
            entries.append({
                'idx':        total,
                'name':       name_py,
                'coord':      coord_py,
                'noisy_label': noisy_py,
                'pred_label':  pred_py,
                'conf':        conf_py,
                'unc':         unc_py,
            })
            total += 1
            
    # --------------------------
    # Within-class selection + high-confidence correction (dataset level)
    # --------------------------

    by_class = defaultdict(list)
    for e in entries:
        by_class[e['noisy_label']].append(e['idx'])

    keep_flags = [False] * total
    final_labels = [None] * total
    kept_count, corrected_count = 0, 0

    for cls, idx_list in by_class.items():
        if len(idx_list) == 0:
            continue
        # 该类内按不确定度升序排序（低不确定度优先）
        idx_sorted = sorted(idx_list, key=lambda i: entries[i]['unc'])
        n_keep = max(0, math.ceil(keep_ratio_per_class * len(idx_sorted)))
        keep_set = set(idx_sorted[:n_keep])

        for i in idx_sorted:
            if i in keep_set:
                keep_flags[i] = True
                kept_count += 1
                # 纠正：高置信则用预测标签，否则保留 noisy_label
                if entries[i]['conf'] >= correct_conf_thres:
                    final_labels[i] = entries[i]['pred_label']
                    corrected_count += 1
                else:
                    final_labels[i] = entries[i]['noisy_label']
            else:
                keep_flags[i] = False
                final_labels[i] = entries[i]['noisy_label']   

    # --------------------------
    # stage2： 
    # --------------------------
    canvases: Dict[str, Dict[str, torch.Tensor]] = {}
    iter_idx = 0  # 对齐 entries 的全局顺序
    for batch in test_loader:
        x, sam_masks, sam_psd_labels, sam_gt_labels, names, coordinates = batch

         
        if sam_masks.dim() == 3:
            sam_masks = sam_masks.unsqueeze(1)
        sam_masks = sam_masks.float()
        B = x.size(0)

        for i in range(B):
            e = entries[iter_idx]
            iter_idx += 1
            if not keep_flags[e['idx']]:
                continue   

            name = e['name']
            x0, y0, w, h = [int(v) for v in e['coord']]
            # 边界裁剪
            x0 = max(0, min(W0, x0)); y0 = max(0, min(H0, y0))
            w  = max(0, min(W0 - x0, w)); h = max(0, min(H0 - y0, h))
            if w == 0 or h == 0:
                continue

             
            m = sam_masks[i:i+1]  # [1,1,Hm,Wm]
            m_resized = F.interpolate(m, size=(h, w), mode='nearest')     # [1,1,h,w]
            m_resized = (m_resized > 0.5).squeeze().to(torch.bool)        # [h,w]
            if not m_resized.any():
                continue

            # 画布初始化
            if name not in canvases:
                canvases[name] = {
                    'label': torch.full((H0, W0), 255, dtype=torch.long),         
                    'conf':  torch.zeros((H0, W0), dtype=torch.float32)
                }

            # 目标区域视图
            y1, y2 = y0, y0 + h
            x1, x2 = x0, x0 + w
            region_lbl = canvases[name]['label'][y1:y2, x1:x2]  # [h,w]
            region_cf  = canvases[name]['conf'][y1:y2, x1:x2]   # [h,w]

            # 最终标签与该样本置信度
            lbl = int(final_labels[e['idx']])
            cf  = float(e['conf'])

            # 重叠区域采用置信度优先覆盖
            upd = m_resized & (cf > region_cf)
            if upd.any():
                region_lbl[upd] = lbl
                region_cf[upd]  = cf

    # --------------------------
    # 汇总统计与返回
    # --------------------------
    total_samples = len(entries)
    stats = {
        'total_samples': total_samples,
        'kept_samples': kept_count,
        'filtered_samples': total_samples - kept_count,
        'corrected_samples': corrected_count,
        'keep_ratio_per_class': keep_ratio_per_class,
        'correct_conf_thres': correct_conf_thres,
    }

    # 导出每个样本的最终决策（可用于保存 CSV）
    decisions: List[Dict] = []
    for e in entries:
        decisions.append({
            'name': e['name'],
            'x': int(e['coord'][0]), 'y': int(e['coord'][1]),
            'w': int(e['coord'][2]), 'h': int(e['coord'][3]),
            'noisy_label': e['noisy_label'],
            'pred_label':  e['pred_label'],
            'final_label': final_labels[e['idx']],
            'conf': e['conf'],
            'unc':  e['unc'],
            'kept': keep_flags[e['idx']],
        })

    save_dir_png = "filtered_labels_png"
    os.makedirs(save_dir_png, exist_ok=True)
    
    for name, data in canvases.items():
         
        label_tensor = data['label']   
        label_np = label_tensor.cpu().numpy().astype(np.uint8)   
        
        
        save_path_png = os.path.join(save_dir_png, f"{name}")
        os.makedirs(os.path.dirname(save_path_png), exist_ok=True)
        label_np = get_color_pallete(label_np, "city")
        label_np.save(save_path_png)
    
    return canvases, stats, decisions


@torch.no_grad()
def test(cfg, saveres, mode='inference', top_k_percent=0.8, confidence_threshold=0.95):
    print('test mode', mode)
    logger = logging.getLogger("SCC_Unc.tester")
    logger.info("Start testing")
    device = torch.device(cfg.MODEL.DEVICE)

    feature_extractor = build_feature_extractor(cfg)
    feature_extractor.to(device)

    classifier = build_classifier(cfg)
    classifier.to(device)

    graph_builder = GraphBuilder()
    graph_vae = build_GraphVAE(cfg)
    graph_vae.to(device)
    
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(feature_extractor_weights)
        classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)

    feature_extractor.eval()
    classifier.eval()
    graph_vae.eval()
    
    torch.cuda.empty_cache()  # TODO check if it helps
    dataset_name = cfg.DATASETS.TEST
    output_folder = '.'
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)

    test_data = build_dataset(cfg, mode='test', is_source=False)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, sampler=None)

    filtered_labels = []  
    corrected_labels = []   
    original_labels = []   
    all_uncertainties = []   
    
    for x, sam_masks, sam_psd_labels, sam_gt_labels, name, coordinates in test_loader:
        # print('name', name)
        x = x.to(device)
        sam_masks = sam_masks.to(device).unsqueeze(1)  # [B, 1, H, W]
        coordinates = coordinates.to(device).float()  # [B, 4]

        features = feature_extractor(x, sam_masks)  # [B, C]
        _, sam_feat = classifier(features, sam_masks)
        
        adj, node_feats, _ = GraphBuilder().build_graph(
            features=sam_feat, coordinates=coordinates, history_features=None, history_coords=None
        )

        recon_x, ori_x, mu, logvar, logits = graph_vae(adj, sam_feat.unsqueeze(0), labels=None)

        U, _, _ = graph_vae.compute_uncertainty(
            recon_x, ori_x, logits, mu=mu, logvar= logvar,
            w_rec=1.0, w_ent=1.0, w_kl=1.0, tau=2.0, w_min=0.1, ignore_index=255
        )
        
        unique_labels = torch.unique(sam_psd_labels)  
        for label in unique_labels:
            label_mask = sam_psd_labels == label

            if label_mask.sum() == 0:
                continue
            
            ## U shape
            print('U.shape', U.shape)
            print('label_mask.shape', label_mask.shape)

            category_uncertainty = U[label_mask].mean(dim=-1)  
            sorted_uncertainty, sorted_idx = torch.sort(category_uncertainty)  

            top_k_idx = sorted_idx[:int(top_k_percent * len(sorted_idx))]
            filtered_samples = x[label_mask][top_k_idx]

            p = torch.softmax(logits, dim=-1)  # [B, N, K]
            confidence, predicted_labels = torch.max(p, dim=-1) 

            corrected_samples_idx = confidence[label_mask][top_k_idx] > confidence_threshold
            corrected_labels.extend(predicted_labels[corrected_samples_idx].cpu().numpy())
            original_labels.extend(sam_gt_labels[label_mask][top_k_idx][corrected_samples_idx].cpu().numpy())

            sam_psd_labels[label_mask][top_k_idx][corrected_samples_idx] = predicted_labels[label_mask][top_k_idx][corrected_samples_idx]

            filtered_labels.append(sam_psd_labels[label_mask][top_k_idx].cpu().numpy())

    return filtered_labels, corrected_labels, original_labels


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Testing")
    parser.add_argument("-cfg",
        "--config-file",
        default="configs/segformer_mitbx_src.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument('--saveres', action="store_true",
                        help='save the result')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("SCC_Unc", save_dir, 0)
    logger.info(cfg)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))


    filter_correct_and_assemble(cfg, args.saveres)


if __name__ == "__main__":
    main()

