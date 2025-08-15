



ckpt="results/mitb5_PhotoDistort_diff/model_iter060000.pth"
ckpt='results/mitb5_PhotoDistort_diff_gta/model_iter060000.pth'
ckpt='results/mitb5_PhotoDistort_diff_gta/model_iter045000.pth'




GguID=6

# test cityscapes 50
#CUDA_VISIBLE_DEVICES=1  python test.py -cfg configs/segformer_mitbx_src_slide.yaml resume $ckpt
CUDA_VISIBLE_DEVICES=$GguID  python test.py -cfg configs/segformer_mitbx_src_diff.yaml resume $ckpt

# test acdc 38
CUDA_VISIBLE_DEVICES=$GguID  python test.py -cfg configs/eval_acdc.yaml resume $ckpt
# test bdd 45
CUDA_VISIBLE_DEVICES=$GguID python test.py -cfg configs/eval_bdd.yaml resume $ckpt
# test mapi 55.8
CUDA_VISIBLE_DEVICES=$GguID  python test.py -cfg configs/eval_mapi.yaml resume $ckpt



CUDA_VISIBLE_DEVICES=0 python test_id.py --saveres -cfg configs/eval_acdc.yaml resume ./results/mitb5_PhotoDistort/model_iter024000.pth

CUDA_VISIBLE_DEVICES=0 nohup python test_id.py --saveres -cfg configs/eval_acdc.yaml resume ./results/mitb5_PhotoDistort/model_iter024000.pth > log_PhotoDistort.txt 2>&1 &



