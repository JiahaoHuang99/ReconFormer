
# ReconFormer x8
task_name=ReconFormer_SKMTEA_AF8_2CH
rm log_${task_name}.txt
nohup python -u main_recon_skmtea.py \
--phase train \
--model ReconFormer \
--epochs 50 \
--challenge singlecoil \
--bs 2 \
--F_path /media/NAS03/SKM-TEA/d.0.4.reconformer \
--train_dataset F \
--test_dataset F \
--sequence default \
--accelerations 8 \
--center-fractions 0.04 \
--lr 0.0002 \
--lr-step-size 5 \
--lr-gamma 0.9 \
--save_dir /home/jh/ReconFormer/runs/${task_name} \
--verbose \
--gpu 0 1 \
>> log_${task_name}.txt &


# ReconFormer x16 (AYL14)
#task_name=ReconFormer_SKMTEA_AF16_2CH
#rm log_${task_name}.txt
#nohup python -u main_recon_skmtea.py \
#--phase train \
#--model ReconFormer \
#--epochs 50 \
#--challenge singlecoil \
#--bs 2 \
#--F_path /media/NAS03/SKM-TEA/d.0.4.reconformer \
#--train_dataset F \
#--test_dataset F \
#--sequence default \
#--accelerations 16 \
#--center-fractions 0.02 \
#--lr 0.0002 \
#--lr-step-size 5 \
#--lr-gamma 0.9 \
#--save_dir /home/jh/ReconFormer/runs/${task_name} \
#--verbose \
#--gpu 1 \
#>> log_${task_name}.txt &
#




# DEBUG
#--phase train
#--model ReconFormer
#--epochs 50
#--challenge singlecoil
#--bs 2
#--F_path /media/NAS03/fastMRI/knee/d.4.0.reconformer.sc
#--train_dataset F
#--test_dataset F
#--sequence default
#--accelerations 4
#--center-fractions 0.08
#--lr 0.0002
#--lr-step-size 5
#--lr-gamma 0.9
#--save_dir /home/jh/ReconFormer/runs/DEBUG
#--verbose
#--gpu 0 1
