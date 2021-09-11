python -m torch.distributed.launch --nproc_per_node 1  tools/test_net.py  --config-file configs/glide/hsrc2016_proj.yaml --ckpt /home/sdc/zwl/gliding_vertex/exp_hsrc2016/hsrc2016_1/model_final.pth
