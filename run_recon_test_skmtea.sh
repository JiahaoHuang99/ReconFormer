
# ReconFormer x8
task_name=ReconFormer_SKMTEA_AF8_2CH
rm log_test_${task_name}.txt
nohup python -u main_recon_test_skmtea.py \
--phase test \
--model ReconFormer \
--challenge singlecoil \
--F_path /media/NAS03/SKM-TEA/d.0.4.reconformer \
--test_dataset F \
--sequence default \
--accelerations 8 \
--center-fractions 0.04 \
--checkpoint /media/NAS04/jiahao/ReconFormer/SKMTEA/runs/ReconFormer_SKMTEA_AF8_2CH/2_net.pth \
--gpu 0 \
--verbose \
>> log_test_${task_name}.txt &


# ReconFormer x16
task_name=ReconFormer_SKMTEA_AF16_2CH
rm log_test_${task_name}.txt
nohup python -u main_recon_test_skmtea.py \
--phase test \
--model ReconFormer \
--challenge singlecoil \
--F_path /media/NAS03/SKM-TEA/d.0.4.reconformer \
--test_dataset F \
--sequence default \
--accelerations 16 \
--center-fractions 0.02 \
--checkpoint /media/NAS04/jiahao/ReconFormer/SKMTEA/runs/ReconFormer_SKMTEA_AF16_2CH/2_net.pth \
--gpu 1 \
--verbose \
>> log_test_${task_name}.txt &


# DEBUG
#--phase test
#--model ReconFormer
#--challenge singlecoil
#--F_path /media/NAS03/SKM-TEA/d.0.4.reconformer
#--test_dataset F
#--sequence default
#--accelerations 8
#--center-fractions 0.04
#--checkpoint /media/NAS04/jiahao/ReconFormer/SKMTEA/runs/ReconFormer_SKMTEA_AF8_2CH/2_net.pth
#--gpu 0
#--verbose