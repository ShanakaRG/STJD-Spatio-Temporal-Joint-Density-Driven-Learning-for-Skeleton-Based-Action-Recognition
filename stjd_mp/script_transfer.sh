export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=2,3

# mamp
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/pkuv2_xsub_joint/finetune_t120_layer8_decay.yaml \
--output_dir ./output_dir/pkuv2_xsub_joint/transfer_from_ntu60_to_pkuV2_stjdmp \
--log_dir ./output_dir/pkuv2_xsub_joint/transfer_from_ntu60_to_pkuV2_stjdmp \
--finetune ./output_dir/ntu60_xsub_joint/pretrain_STJDMP_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5

python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/pkuv2_xsub_joint/finetune_t120_layer8_decay.yaml \
--output_dir ./output_dir/pkuv2_xsub_joint/transfer_from_ntu120_to_pkuV2_stjdmp \
--log_dir ./output_dir/pkuv2_xsub_joint/transfer_from_ntu120_to_pkuV2_stjdmp \
--finetune ./output_dir/ntu120_xsub_joint/pretrain_STJDMP_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5

python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/pkuv2_xsub_joint/finetune_t120_layer8_decay.yaml \
--output_dir ./output_dir/pkuv2_xsub_joint/transfer_from_pkuv1_to_pkuV2_stjdmp \
--log_dir ./output_dir/pkuv2_xsub_joint/transfer_from_pkuv1_to_pkuV2_stjdmp \
--finetune ./output_dir/pkuv1_xsub_joint/pretrain_STJDMP_ep400_noamp/checkpoint-300.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5

