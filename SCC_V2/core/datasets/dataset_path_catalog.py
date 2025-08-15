import os
from .cityscapes_cls import CityscapesCLSDataSet

class DatasetCatalog(object):
    DATASETS = {
        "G2C_train": {
            "data_dir": "/data/zd/SeCoV2/data/UDA/cityscapes/train",
            "data_list": "/data/zd/SeCoV2/data/UDA/G2C/DTST_PL/CS_no_thr/train_sam_cls_mask_psa_infos_cls.json"
        },
        "G2C_test": {
            "data_dir": "G2C/DTST_PL/CS_no_thr",
            "data_list": "train_sam_mask_infos.json"
        },
        "C2B_train": {
            "data_dir": "/data/zd/SeCoV2/data/UDA/BDD/train",
            "data_list": "/data/zd/SeCoV2/data/UDA/C2B/HRDA/BDD_no_thr/train_sam_cls_mask_psa_infos_cls.json"
        },
        "C2B_test": {
            "data_dir": "/data/zd/SeCoV2/data/UDA/BDD/train",
            "data_list": "train_sam_mask_infos.json"
        },    
        "G2CDSD_train": {
            "data_dir": "/data/zd/SeCoV2/data/UDA/CDSD/train",
            "data_list": "/data/zd/SeCoV2/data/UDA/C2B/MIC/CDSD_no_thr/train_sam_cls_mask_psa_infos_cls.json"
        },
        "G2CD-SD_test": {
            "data_dir": "/data/zd/SeCoV2/data/UDA/CDSD/train",
            "data_list": "train_sam_mask_infos.json"
        },    
        "USA2Sing_train": {
            "data_dir": "/data/zd/SeCoV2/data/UDA/Sing/train",
            "data_list": "/data/zd/SeCoV2/data/UDA/C2B/MMDA/Sing_no_thr/train_sam_cls_mask_psa_infos_cls.json"
        },
        "USA2S_test": {
            "data_dir": "/data/zd/SeCoV2/data/UDA/Sing/train",
            "data_list": "train_sam_mask_infos.json"
        },    
    }

    @staticmethod
    def get(name, mode, num_classes, max_iters=None, transform=None, rsc=None, cfg=None):
        if "cityscapes" in name:
            args = DatasetCatalog.DATASETS[name]
            return CityscapesCLSDataSet(args["data_dir"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
        
        raise RuntimeError("Dataset not available: {}".format(name))



