#! /bash/bin
# accelerate launch train_edlora.py -opt /home/user/R13943013/dlcv_final/Mix-of-Show/options/train/EDLoRA/real/8101_EDLoRA_Q_bow.yml
export NCCL_IB_DISABLE="1"
export NCCL_P2P_DISABLE="1"
export CCUDA_VISIBLE_DEVICES="0"
# prompt list
# 1. cat2.yml
# 2. dog6.yml
# 3. flower1.yml
# 4. dog.yml
# 5. pet_cat1.yml
# 6. vase.yml
# 7. watercolor.yml
# 8. wearable_glasses.yml
# usage: python3 train_edlora.py -opt <prompt.yml>
train_yaml=$1
python3  train_edlora.py -opt ./options/$train_yaml