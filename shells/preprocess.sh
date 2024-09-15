# step0 make df for prepare and train
python ./preprocess/step0_make_df.py

# step1 patch
nohup python ./preprocess/step1_Patching.py --input_csv ./out_csv/your data fram.csv --output_folder ./your save path to patch dir

# step2 extract features
nohup python ./preprocess/step2_extract_features.py --base_dir . --patch_dir ./your patch dir --pretrained_ckpt ./checkpoints --save_dir_p8 your save path to patch 8 --save_dir_p16 your save path to patch 16 > ./preprocess/logs/patch.log 2>&1 &


