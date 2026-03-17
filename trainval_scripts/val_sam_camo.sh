python scripts/evaluate_model_sam.py SAM\
  --model_dir=/home/vilab/khj/ssd5/colseg/sam_checkpoints/\
  --checkpoint=sam_vit_b_01ec64.pth\
  --infer-size=256\
  --datasets=CAMO\
  --gpus=0\
  --n-clicks=20\
  --target-iou=0.85\
  --thresh=0.50\
  --vis
  #--target-iou=0.95\
# sam_vit_b_01ec64.pth
#--datasets=GrabCut,Berkeley,DAVIS\ last_checkpoint
#--datasets=DAVIS_high,DAVIS_mid,DAVIS_low\
  

