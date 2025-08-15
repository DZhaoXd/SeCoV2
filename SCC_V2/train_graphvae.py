import argparse
import os
import datetime
import logging
import time
import math
import numpy as np

import torch
import torch.nn.functional as F

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_model, build_feature_extractor, build_classifier, build_GraphVAE, GraphBuilder
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
import warnings
import random
from torch.backends import cudnn
from tqdm import tqdm   
from torch.utils.data import DataLoader, WeightedRandomSampler



warnings.filterwarnings('ignore')

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


def amp_backward(loss, optimizer, retain_graph=False):
    if APEX_AVAILABLE:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(retain_graph=retain_graph)
    else:
        loss.backward(retain_graph=retain_graph)


def strip_prefix_if_present(state_dict, prefix):
    from collections import OrderedDict
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix + 'layer5'):
            continue
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def train(cfg, local_rank, distributed):
    logger = logging.getLogger("SCC_Unc.trainer")
    logger.info("Start training")

    feature_extractor = build_feature_extractor(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor.to(device)

    classifier = build_classifier(cfg)
    classifier.to(device)
    
    graph_builder = GraphBuilder()
    GraphVAE = build_GraphVAE(cfg)
    GraphVAE.to(device)
    
    # if local_rank == 0:
    #     print(feature_extractor)
    #     print(classifier)

    model_name, backbone_name = cfg.MODEL.NAME.split('_')

    batch_size = cfg.SOLVER.BATCH_SIZE
    if distributed:
        pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))

        batch_size = int(cfg.SOLVER.BATCH_SIZE / torch.distributed.get_world_size())
        # if not cfg.MODEL.FREEZE_BN:
        #     feature_extractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_extractor)
        feature_extractor = torch.nn.parallel.DistributedDataParallel(
            feature_extractor, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg1
        )
        pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg2
        )
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()

    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()

    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR * 10, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()

    optimizer_vae = torch.optim.SGD(GraphVAE.parameters(), lr=cfg.SOLVER.BASE_LR * 10, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_vae.zero_grad()
    
    
    output_dir = cfg.OUTPUT_DIR

    save_to_disk = local_rank == 0

    iteration = 0

    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        model_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(
            checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(model_weights)
        classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(
            checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)

    src_train_data = build_dataset(cfg, mode='train', is_source=True)
     
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(src_train_data)
    else:
        num_samples = cfg.SOLVER.MAX_ITER*cfg.SOLVER.BATCH_SIZE
        train_sampler = WeightedRandomSampler(
                        weights=src_train_data.sample_weights,
                        num_samples=num_samples,
                        replacement=True
                        )

    train_loader = torch.utils.data.DataLoader(
        src_train_data,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    
    max_iters = cfg.SOLVER.MAX_ITER
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    feature_extractor.train()
    classifier.train()
    GraphVAE.train()
    start_training_time = time.time()
    end = time.time()
    
    if APEX_AVAILABLE:
        [feature_extractor, classifier, GraphVAE], [optimizer_fea, optimizer_cls, optimizer_vae] = amp.initialize(
           [feature_extractor, classifier, GraphVAE], [optimizer_fea,optimizer_cls, optimizer_vae],  opt_level="O2", keep_batchnorm_fp32=True,
             loss_scale="dynamic"
        )  
    
    
    
    for i, (src_input, sam_masks, sam_psd_labels, sam_gt_labels, _, coordinates) in enumerate(train_loader):
        data_time = time.time() - end
        current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters,
                                          power=cfg.SOLVER.LR_POWER)
        for index in range(len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[index]['lr'] = current_lr * 10
        for index in range(len(optimizer_vae.param_groups)):
            optimizer_vae.param_groups[index]['lr'] = current_lr * 10
            
        # Zero the gradients
        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()
        optimizer_vae.zero_grad()
        
        # Move data to GPU
        src_input = src_input.cuda(non_blocking=True)
        sam_masks = sam_masks.cuda(non_blocking=True).unsqueeze(1)
        # unique_labels = torch.unique(seg_labels)
        # print("Unique labels in seg_labels:", unique_labels)
        sam_psd_labels = sam_psd_labels.cuda(non_blocking=True).long()
        sam_gt_labels = sam_gt_labels.cuda(non_blocking=True).long()
        coordinates = coordinates.cuda(non_blocking=True)
        
        # Feature extraction and classifier input adjustments        
        features = feature_extractor(src_input, sam_masks)
        # Use SAMClassifier for region classification
        _, sam_feat = classifier(features, sam_masks)
        
        with torch.no_grad():
            adj_matrix, node_feats, _ = graph_builder.build_graph(sam_feat, coordinates)    
        sam_feat = sam_feat.unsqueeze(0)   
        recon_features, ori_features, mu, logvar, logits = GraphVAE(adj_matrix, sam_feat)

        # loss    
        sam_psd_labels = sam_psd_labels.view(-1)  # [B * num_regions]
        sam_gt_labels = sam_gt_labels.view(-1)  # [B * num_regions]
        sam_loss = GraphVAE.combined_loss(recon_features, ori_features, mu, logvar, logits, sam_psd_labels, iteration, cfg.SOLVER.STOP_ITER )
            

        loss, loss_logs = sam_loss 
        # loss.backward()
        amp_backward(loss, [optimizer_fea, optimizer_cls, optimizer_vae])
        
        optimizer_fea.step()
        optimizer_cls.step()
        optimizer_vae.step()
        meters.update(ce=loss_logs['ce'].item())
        meters.update(rec=loss_logs['rec'].item())
        meters.update(kl=loss_logs['beta_kl'].item())
        iteration += 1

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iters - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        if iteration % 20 == 0 or iteration == max_iters:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer_fea.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if (iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or iteration == cfg.SOLVER.STOP_ITER):
            filename = os.path.join(output_dir, "model_iter{:06d}.pth".format(iteration))
            torch.save({'iteration': iteration, 'feature_extractor': feature_extractor.state_dict(),
                        'classifier': classifier.state_dict(), }, filename)
            run_test(cfg, (feature_extractor, classifier, GraphVAE), local_rank, distributed)

        if iteration == max_iters:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iters)
        )
    )

    return feature_extractor, classifier


def run_test(cfg, model, local_rank, distributed):
    logger = logging.getLogger("SCC_Unc.trainer")
    if local_rank == 0:
        logger.info('>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    conn_acc_meter = AverageMeter()  
    class_acc_meter = {i: AverageMeter() for i in range(cfg.MODEL.NUM_CLASSES)}  

    feature_extractor, classifier, graphVAE = model

    if distributed:
        feature_extractor, classifier = feature_extractor.module, classifier.module
    torch.cuda.empty_cache()
    dataset_name = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)

    test_data = build_dataset(cfg, mode='test', is_source=False)
    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4,
                                              pin_memory=True, sampler=test_sampler)
    feature_extractor.eval()
    classifier.eval()
    end = time.time()
    graph_builder = GraphBuilder()
    with torch.no_grad():
        for i, (x, sam_masks, sam_psd_labels, sam_gt_labels, _, coordinates) in enumerate(tqdm(test_loader, desc="Testing", ncols=100)):
            x = x.cuda(non_blocking=True)
            sam_masks = sam_masks.cuda(non_blocking=True).unsqueeze(1)
            sam_gt_labels = sam_gt_labels.cuda(non_blocking=True).long()
            coordinates = coordinates.cuda(non_blocking=True).long()
    
            features = feature_extractor(x, sam_masks)
            _, sam_feat = classifier(features, sam_masks)
            adj_matrix, node_feats, _ = graph_builder.build_graph(sam_feat, coordinates)
            recon_features, ori_features, mu, logvar, logits = graphVAE(adj_matrix, sam_feat.unsqueeze(0))
    
            num_classes = cfg.MODEL.NUM_CLASSES
            logits = logits.reshape(-1, num_classes)          
            pred = logits.argmax(dim=1)                       # [N]
    
            gt = sam_gt_labels.view(-1)                       # [N]
            valid = (gt != 255)
    
            
            correct = (pred == gt) & valid
            num_correct = correct.sum().item()
            num_valid   = valid.sum().item()
            if num_valid > 0:
                conn_acc_meter.update(num_correct / num_valid, n=num_valid) 
    
            
            for c in range(num_classes):
                class_mask = (gt == c) & valid
                total_c = class_mask.sum().item()
                if total_c > 0:
                    correct_c = (pred[class_mask] == gt[class_mask]).sum().item()
                    class_acc_meter[c].update(correct_c / total_c, n=total_c)
    
            batch_time.update(time.time() - end)
            end = time.time()
    
    
    conn_acc = conn_acc_meter.avg 
    
    logger.info('Testing Results:')
    logger.info(f'Overall Accuracy (micro): {conn_acc:.4f}')
    
   
    class_accs, appeared = [], 0
    for i in range(cfg.MODEL.NUM_CLASSES):
        acc_i = class_acc_meter[i].avg
        if class_acc_meter[i].count > 0:   
            appeared += 1
            class_accs.append(acc_i)
        logger.info(f'Class_{i} Conn Accuracy: {acc_i:.4f}')
    
    if appeared > 0:
        macro_acc = sum(class_accs) / appeared
        logger.info(f'Macro Accuracy (over appeared classes): {macro_acc:.4f}')


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="configs/segformer_mitbx_src.yaml",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--seed", type=int, default=8888)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    #torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # os.environ['WORLD_SIZE'] = '2'
    # os.environ['RANK'] = '0'
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # a=int(os.environ["WORLD_SIZE"])
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        # RANK = int(os.environ["RANK"])
        # if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        #     NGPUS_PER_NODE = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        # else:
        #     NGPUS_PER_NODE = torch.cuda.device_count()
        # assert NGPUS_PER_NODE > 0, "CUDA is not supported"
        # GPU = RANK % NGPUS_PER_NODE
        # torch.cuda.set_device(GPU)
        # master_address = os.environ['MASTER_ADDR']
        # master_port = int(os.environ['MASTER_PORT'])
        # WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        # torch.distributed.init_process_group(backend='nccl',
        #                                      init_method='tcp://{}:{}'.format(
        #                                          master_address, master_port),
        #                                      rank=RANK, world_size=WORLD_SIZE)
        # NUM_GPUS = WORLD_SIZE
        # print(f"RANK and WORLD_SIZE in environ: {RANK}/{WORLD_SIZE}")

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("SCC_Unc.trainer", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, model, args.local_rank, args.distributed)


if __name__ == "__main__":
    main()
