export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=5,6


######## NTU60 xsub

for((i=1;i<=50;i++)); 
do
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12343 main_finetune.py \
--config ./config/ntu60_xsub_joint/semi_0.01_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xsub_joint/semi001/semi_0.01_stjdmp_{$i} \
--log_dir ./output_dir/ntu60_xsub_joint/semi001/semi_0.01_stjdmp_{$i} \
--finetune ./output_dir/ntu60_xsub_joint/pretrain_STJDMP_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5
done

for((i=1;i<=15;i++)); 
do
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12341 main_finetune.py \
--config ./config/ntu60_xsub_joint/semi_0.1_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xsub_joint/semi01/semi_0.1_stjdmp_{$i} \
--log_dir ./output_dir/ntu60_xsub_joint/semi01/semi_0.1_stjdmp_{$i} \
--finetune ./output_dir/ntu60_xsub_joint/pretrain_STJDMP_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5
done



######## NTU60 xview

for((i=1;i<=50;i++)); 
do
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12347 main_finetune.py \
--config ./config/ntu60_xview_joint/semi_0.01_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xview_joint/semi001/semi_0.01_stjdmp_{$i} \
--log_dir ./output_dir/ntu60_xview_joint/semi001/semi_0.01_stjdmp_{$i} \
--finetune ./output_dir/ntu60_xview_joint/pretrain_STJDMP_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5
done

for((i=1;i<=15;i++)); 
do
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60_xview_joint/semi_0.1_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xview_joint/semi01/semi_0.1_stjdmp_{$i} \
--log_dir ./output_dir/ntu60_xview_joint/semi01/semi_0.1_stjdmp_{$i} \
--finetune ./output_dir/ntu60_xview_joint/pretrain_STJDMP_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5
done


