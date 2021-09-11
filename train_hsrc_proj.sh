#python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --config-file configs/glide/hsrc2016_proj.yaml
CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1  tools/test_net.py  --config-file configs/glide/hsrc2016_proj.yaml --ckpt exp_hsrc2016/hsrc2016_proj_exer1_init_dataaug/model_0030000.pth 
#CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1  tools/test_net.py  --config-file configs/glide/hsrc2016_proj.yaml --ckpt exp_hsrc2016/hsrc2016_proj_exer1_init_dataaug/model_0050000.pth
