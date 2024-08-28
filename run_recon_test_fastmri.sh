
# ReconFormer x4
task_name=ReconFormer_FastMRI_AF4_2CH
rm log_test_${task_name}.txt
nohup python -u main_recon_test_fastmri.py \
--phase test \
--model ReconFormer \
--challenge singlecoil \
--F_path /media/NAS03/fastMRI/knee/d.4.0.reconformer.sc \
--test_dataset F \
--sequence PD \
--accelerations 4 \
--center-fractions 0.08 \
--checkpoint /media/NAS04/jiahao/ReconFormer/FastMRI/runs/ReconFormer_FastMRI_AF4_2CH/2_net.pth \
--gpu 0 \
--verbose \
>> log_test_${task_name}.txt &


# ReconFormer x8
task_name=ReconFormer_FastMRI_AF8_2CH
rm log_test_${task_name}.txt
nohup python -u main_recon_test_fastmri.py \
--phase test \
--model ReconFormer \
--challenge singlecoil \
--F_path /media/NAS03/fastMRI/knee/d.4.0.reconformer.sc \
--test_dataset F \
--sequence PD \
--accelerations 8 \
--center-fractions 0.04 \
--checkpoint /media/NAS04/jiahao/ReconFormer/FastMRI/runs/ReconFormer_FastMRI_AF8_2CH/3_net.pth \
--gpu 1 \
--verbose \
>> log_test_${task_name}.txt &


# DEBUG
#--phase test
#--model ReconFormer
#--challenge singlecoil
#--F_path /media/NAS03/fastMRI/knee/d.4.0.reconformer.sc
#--test_dataset F
#--sequence PD
#--accelerations 8
#--center-fractions 0.04
#--checkpoint /media/NAS04/jiahao/ReconFormer/FastMRI/runs/ReconFormer_FastMRI_AF8_2CH/3_net.pth
#--gpu 0
#--verbose