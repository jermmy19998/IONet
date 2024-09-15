# 1. train abmil patch size 16
nohup python ./train.py \
    --base_dir . \
    --train_csv ./sc_train_40x.csv \
    --test_csv ./sc_test_40x.csv \
    --save_ckpt_dir ./ckpt/ \
    --mil_type abmil \
    --num_worker 64 \
    --epoch 20 \
    --device cuda:1 \
    --patch_size 16 > ./logs/train_ab_p16.log 2>&1 &


# 2. train abmil patch size 8
nohup python ./train.py \
    --base_dir . \
    --train_csv ./sc_train_40x.csv \
    --test_csv ./sc_test_40x.csv \
    --save_ckpt_dir ./ckpt/weighted \
    --mil_type abmil \
    --num_worker 64 \
    --epoch 20 \
    --device cuda:2 \
    --patch_size 8 > ./logs/train_ab_p8.log 2>&1 &


# 3. train dsmil patch size 8
nohup python ./train.py \
    --base_dir . \
    --train_csv ./sc_train_40x.csv \
    --test_csv ./sc_test_40x.csv \
    --save_ckpt_dir ./ckpt/ \
    --mil_type dsmil \
    --num_worker 64 \
    --epoch 20 \
    --device cuda:3 \
    --patch_size 8 > ./logs/train_ds_p8.log 2>&1 &


# 4. train dsmil patch size 16
nohup python ./train.py \
    --base_dir . \
    --train_csv ./sc_train_40x.csv \
    --test_csv ./sc_test_40x.csv \
    --save_ckpt_dir ./ckpt/ \
    --mil_type dsmil \
    --num_worker 64 \
    --epoch 20 \
    --device cuda:3 \
    --patch_size 16 > ./logs/train_ds_p16.log 2>&1 &

