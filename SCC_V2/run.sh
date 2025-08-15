
CUDA_VISIBLE_DEVICES=1 python train_src.py -cfg configs/SCC_Unc.yaml OUTPUT_DIR results/SCC_Unc

CUDA_VISIBLE_DEVICES=1 nohup python train_src.py -cfg configs/SCC_Unc.yaml OUTPUT_DIR results/SCC_Unc > logs/train_gt.log  2>&1 &