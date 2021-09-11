save_dir='dota_proj_ms/exer4_ML_ADP_init2_aug_weight01_lr002'
#CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 tools/train_net.py --config-file configs/glide/dota_proj.yaml
#CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1  tools/test_net.py  --config-file configs/glide/dota_proj.yaml --ckpt /SDB/zwl/gliding_vertex_proj/exp_dota/${save_dir}/model_best.pth
for ((model=30000;model<=60000;model=model+5000))
do
{
gpu=$(((model-30000)/5000))
#CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch --nproc_per_node=1  tools/test_net.py  --config-file configs/glide/dota_proj.yaml --ckpt /SDB/zwl/gliding_vertex_proj/exp_dota/${save_dir}/model_00${model}.pth

model2=$((model+5000))
gpu=$(((model2-35000)/10000))
#num=3
#if ((${gpu}>${num}));then
#  continue
#fi
#CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch --nproc_per_node=1  tools/test_net.py  --config-file configs/glide/dota_proj.yaml --ckpt /SDB/zwl/gliding_vertex_proj/exp_dota/${save_dir}/model_00${model2}.pth

model2=$((model+2500))
gpu=$(((model2-32500)/10000))
#num=3
#if ((${gpu}>${num}));then
#  continue
#fi
#CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch --nproc_per_node=1  tools/test_net.py  --config-file configs/glide/dota_proj.yaml --ckpt /SDB/zwl/gliding_vertex_proj/exp_dota/${save_dir}/model_00${model2}.pth
}&
done
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1  tools/test_net.py  --config-file configs/glide/dota_proj.yaml --ckpt /SDB/zwl/gliding_vertex_proj/exp_dota/${save_dir}/model_0055000.pth

exit
for ((model=27500;model<=50000;model=model+2500))
do
{
  
CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch --nproc_per_node=1  tools/test_net.py  --config-file configs/glide/dota_proj.yaml --ckpt /SDB/zwl/gliding_vertex_proj/exp_dota/${save_dir}/model_00${model}.pth   gpu=(model-27500)/2500

}&
done
cd ./maskrcnn_benchmark/DOTA_devkit
for ((model=30000;model<=30000;model=model+2500))
do
{
 run="python ResultMerge_v.py --model 00${model} --split trainval --save_dir ${save_dir}"
 ${run}
}&
python ResultMerge_v.py --model final  --split trainval --save_dir ${save_dir}

exit
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1  tools/test_net.py  --config-file configs/glide/dota_proj.yaml --ckpt /SDB/zwl/gliding_vertex_proj/exp_dota/dota_proj_exer1_aug_lr2/model_0027500.pth
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  tools/test_net.py  --config-file configs/glide/dota_proj.yaml --ckpt /SDB/zwl/gliding_vertex_proj/exp_dota/dota_proj_exer1_aug_lr2/model_0030000.pth
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  tools/test_net.py  --config-file configs/glide/dota_proj.yaml --ckpt /SDB/zwl/gliding_vertex_proj/exp_dota/dota_proj_exer1_aug_lr2/model_0032500.pth
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  tools/test_net.py  --config-file configs/glide/dota_proj.yaml --ckpt /SDB/zwl/gliding_vertex_proj/exp_dota/dota_proj_exer1_aug_lr2/model_0035000.pth
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  tools/test_net.py  --config-file configs/glide/dota_proj.yaml --ckpt /SDB/zwl/gliding_vertex_proj/exp_dota/dota_proj_exer1_aug_lr2/model_0037500.pth
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  tools/test_net.py  --config-file configs/glide/dota_proj.yaml --ckpt /SDB/zwl/gliding_vertex_proj/exp_dota/dota_proj_exer1_aug_lr2/model_0040000.pth
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  tools/test_net.py  --config-file configs/glide/dota_proj.yaml --ckpt /SDB/zwl/gliding_vertex_proj/exp_dota/dota_proj_exer1_aug_lr2/model_0042500.pth
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  tools/test_net.py  --config-file configs/glide/dota_proj.yaml --ckpt /SDB/zwl/gliding_vertex_proj/exp_dota/dota_proj_exer1_aug_lr2/model_0035000.pth
