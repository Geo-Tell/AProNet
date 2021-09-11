while ((1))
do
python -m torch.distributed.launch --nproc_per_node=1 tools/train_net.py --config-file configs/glide/hsrc2016.yaml
done
