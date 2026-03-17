python scripts/evaluate_model_sam.py SAM\
  --model_dir=/mnt/jihun2/SAM_ckpt/\
  --checkpoint=sam_vit_h_4b8939.pth\
  --infer-size=256\
  --datasets=COD10k\
  --gpus=0\
  --n-clicks=20\
  --target-iou=0.90\
  --thresh=0.50\
  #--vis
  #--target-iou=0.95\

#--datasets=GrabCut,Berkeley,DAVIS\ last_checkpoint
#--datasets=DAVIS_high,DAVIS_mid,DAVIS_low\
  

