export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NTU60 xsub
python -m torch.distributed.launch --nproc_per_node=4 --master_port 11231 main_pretrain.py \
--config ./config/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_mask90.yaml \
--output_dir ./output_dir/ntu60_xsub_joint/pretrain_STJDMP_ep400_noamp \
--log_dir ./output_dir/ntu60_xsub_joint/pretrain_STJDMP_ep400_noamp \
--weight_path ./weights/ntu60_xsub.pth

# # NTU60 xview
python -m torch.distributed.launch --nproc_per_node=4 --master_port 11232 main_pretrain.py \
--config ./config/ntu60_xview_joint/pretrain_stjd_t120_layer8+3_mask90.yaml \
--output_dir ./output_dir/ntu60_xview_joint/pretrain_STJDMP_ep400_noamp \
--log_dir ./output_dir/ntu60_xview_joint/pretrain_STJDMP_ep400_noamp \
--weight_path ./weights/ntu60_xview.pth

# # NTU120 xset
python -m torch.distributed.launch --nproc_per_node=4 --master_port 11234 main_pretrain.py \
--config ./config/ntu120_xset_joint/pretrain_mamp_t120_layer8+5_mask90.yaml \
--output_dir ./output_dir/ntu120_xset_joint/pretrain_STJDMP_ep400_noamp \
--log_dir ./output_dir/ntu120_xset_joint/pretrain_STJDMP_ep400_noamp \
--weight_path ./weights/ntu120_xset.pth

# # NTU120 xsub
python -m torch.distributed.launch --nproc_per_node=4 --master_port 11235 main_pretrain.py \
--config ./config/ntu120_xsub_joint/pretrain_mamp_t120_layer8+5_mask90.yaml \
--output_dir ./output_dir/ntu120_xsub_joint/pretrain_STJDMP_ep400_noamp \
--log_dir ./output_dir/ntu120_xsub_joint/pretrain_STJDMP_ep400_noamp \
--weight_path ./weights/ntu120_xsub.pth


# # PKU v1
python -m torch.distributed.launch --nproc_per_node=4 --master_port 11234 main_pretrain.py \
--config ./config/pkuv1_xsub_joint/pretrain_mamp_t120_layer8+5_mask90.yaml \
--output_dir ./output_dir/pkuv1_xsub_joint/pretrain_STJDMP_ep400_noamp \
--log_dir ./output_dir/pkuv1_xsub_joint/pretrain_STJDMP_ep400_noamp \
--weight_path ./weights/pku1_xsub.pth

# # PKU v2
python -m torch.distributed.launch --nproc_per_node=4 --master_port 11234 main_pretrain.py \
--config ./config/pkuv2_xsub_joint/pretrain_mamp_t120_layer8+5_mask90.yaml \
--output_dir ./output_dir/pkuv2_xsub_joint/pretrain_STJDMP_ep400_noamp \
--log_dir ./output_dir/pkuv2_xsub_joint/pretrain_STJDMP_ep400_noamp \
--weight_path ./weights/pku2_xsub.pth