
# ReconFormer x4
#task_name=ReconFormer_AF4_2CH
#rm log_${task_name}.txt
#nohup python -u main_recon_fastmri.py \
#--phase train \
#--model ReconFormer \
#--epochs 50 \
#--challenge singlecoil \
#--bs 4 \
#--F_path /media/NAS03/fastMRI/knee/d.4.0.reconformer.sc \
#--train_dataset F \
#--test_dataset F \
#--sequence PD \
#--accelerations 4 \
#--center-fractions 0.08 \
#--lr 0.0002 \
#--lr-step-size 5 \
#--lr-gamma 0.9 \
#--save_dir /home/jh/ReconFormer/runs/${task_name} \
#--verbose \
#--gpu 0 1 2 3 \
#>> log_${task_name}.txt &


# ReconFormer x8
task_name=ReconFormer_AF8_2CH
rm log_${task_name}.txt
nohup python -u main_recon_fastmri.py \
--phase train \
--model ReconFormer \
--epochs 50 \
--challenge singlecoil \
--bs 4 \
--F_path /media/NAS03/fastMRI/knee/d.4.0.reconformer.sc \
--train_dataset F \
--test_dataset F \
--sequence PD \
--accelerations 8 \
--center-fractions 0.04 \
--lr 0.0002 \
--lr-step-size 5 \
--lr-gamma 0.9 \
--save_dir /home/jh/ReconFormer/runs/${task_name} \
--verbose \
--gpu 0 1 2 3 \
>> log_${task_name}.txt &



# DEBUG
#--phase train
#--model ReconFormer
#--epochs 50
#--challenge singlecoil
#--bs 4
#--F_path /media/NAS03/fastMRI/knee/d.4.0.reconformer.sc
#--train_dataset F
#--test_dataset F
#--sequence PD
#--accelerations 4
#--center-fractions 0.08
#--lr 0.0002
#--lr-step-size 5
#--lr-gamma 0.9
#--save_dir /home/jh/ReconFormer/runs/DEBUG
#--verbose
#--gpu 0 1 2 3
