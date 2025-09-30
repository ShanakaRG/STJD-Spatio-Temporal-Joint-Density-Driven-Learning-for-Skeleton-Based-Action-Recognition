export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=7,8


# NTU-60 xsub
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_linprobe.py \
--config ./config/ntu60_xsub_joint/linprobe_t120_layer8.yaml \
--output_dir ./output_dir/ntu60_xsub_joint/linear/linear_stjdmp \
--log_dir ./output_dir/ntu60_xsub_joint/linear/linear_stjdmp \
--finetune ./output_dir/ntu60_xsub_joint/pretrain_STJDMP_ep400_noamp/checkpoint-399.pth \
--dist_eval  



# NTU-60 xview
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_linprobe.py \
--config ./config/ntu60_xview_joint/linprobe_t120_layer8.yaml \
--output_dir ./output_dir/ntu60_xview_joint/linear/linear_stjdmp \
--log_dir ./output_dir/ntu60_xview_joint/linear/linear_stjdmp \
--finetune ./output_dir/ntu60_xview_joint/pretrain_STJDMP_ep400_noamp/checkpoint-399.pth \
--dist_eval


# NTU-120 xset
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_linprobe.py \
--config ./config/ntu120_xset_joint/linprobe_t120_layer8.yaml \
--output_dir ./output_dir/ntu120_xset_joint/linear/linear_stjdmp \
--log_dir ./output_dir/ntu120_xset_joint/linear/linear_stjdmp \
--finetune  ./output_dir/ntu120_xset_joint/pretrain_STJDMP_ep400_noamp/checkpoint-399.pth \
--dist_eval




# NTU-120 xsub
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12344 main_linprobe.py \
--config ./config/ntu120_xsub_joint/linprobe_t120_layer8.yaml \
--output_dir ./output_dir/ntu120_xsub_joint/linear/linear_stjdmp \
--log_dir ./output_dir/ntu120_xsub_joint/linear/linear_stjdmp \
--finetune  ./output_dir/ntu120_xsub_joint/pretrain_STJDMP_ep400_noamp/checkpoint-399.pth \
--dist_eval




# PKU Phase I
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_linprobe.py \
--config ./config/pkuv1_xsub_joint/linprobe_t120_layer8.yaml \
--output_dir ./output_dir/pkuv1_xsub_joint/linear/linear_stjdmp \
--log_dir ./output_dir/pkuv1_xsub_joint/linear/linear_stjdmp \
--finetune ./output_dir/pkuv1_xsub_joint/pretrain_STJDMP_ep400_noamp/checkpoint-399.pth \
--dist_eval



# PKU Phase II
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_linprobe.py \
--config ./config/pkuv2_xsub_joint/linprobe_t120_layer8.yaml \
--output_dir ./output_dir/pkuv2_xsub_joint/linear/linear_stjdmp \
--log_dir ./output_dir/pkuv2_xsub_joint/linear/linear_stjdmp \
--finetune ./output_dir/pkuv2_xsub_joint/pretrain_STJDMP_ep400_noamp/checkpoint-399.pth \
--dist_eval