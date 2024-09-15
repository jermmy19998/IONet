# 1. infer abmil  202
python ./push_code/src/inference_plot.py \
    --base_dir . \
    --test_csv ./push_code/out_csv/OC_202.csv \
    --p8_feature_dir ./202_40x_ViT_p8_448_0.5 \
    --p16_feature_dir ./202_40x_ViT_p16_448_0.5 \
    --abmil_p8_weight ./ckpt/weighted/abmil8_epoch_7_loss_0.2347_bacc_0.9583.pth \
    --abmil_p16_weight ./ckpt/weighted/abmil16_epoch_7_loss_0.2939_bacc_0.9375.pth \
    --num_worker 64 \
    --remove_label \
    --save_result_df ./push_code/result_df/result_202_abmil_all_patch.csv \
    --infer_p8 \
    --infer_p16 \
    --infer_abmil \
    --abmil_p8 0.5 \
    --abmil_p16 0.5 

# 3. infer dsmil  202
python ./push_code/src/inference_plot.py \
    --base_dir . \
    --test_csv ./push_code/out_csv/OC_202.csv \
    --p8_feature_dir ./202_40x_ViT_p8_448_0.5 \
    --p16_feature_dir ./202_40x_ViT_p16_448_0.5 \
    --dsmil_p8_weight ./ckpt/dsmil8_epoch_15_loss_0.3595_bacc_0.8958.pth \
    --dsmil_p16_weight ./ckpt/dsmil16_epoch_8_loss_0.6246_bacc_0.9583.pth \
    --num_worker 64 \
    --remove_label \
    --save_result_df ./push_code/result_df/result_202_dsmil_all_patch.csv \
    --infer_p8 \
    --infer_p8  \
    --infer_dsmil \
    --dsmil_p8 0.5 \
    --dsmil_p16 0.5



# 5. infer abmil  lc
python ./push_code/src/inference_plot.py \
    --base_dir . \
    --test_csv ./push_code/out_csv/OC_lc.csv \
    --p8_feature_dir ./lc_40x_ViT_p8_448_0.5 \
    --p16_feature_dir ./lc_40x_ViT_p16_448_0.5 \
    --abmil_p8_weight ./ckpt/weighted/abmil8_epoch_7_loss_0.2347_bacc_0.9583.pth \
    --abmil_p16_weight ./ckpt/weighted/abmil16_epoch_7_loss_0.2939_bacc_0.9375.pth \
    --num_worker 64 \
    --remove_label \
    --save_result_df ./push_code/result_df/result_lc_abmil_all_patch.csv \
    --infer_p8 \
    --infer_p16 \
    --infer_abmil \
    --abmil_p8 0.5 \
    --abmil_p16 0.5 



# 7. infer dsmil  lc
python ./push_code/src/inference_plot.py \
    --base_dir . \
    --test_csv ./push_code/out_csv/OC_lc.csv \
    --p8_feature_dir ./lc_40x_ViT_p8_448_0.5 \
    --p16_feature_dir ./lc_40x_ViT_p16_448_0.5 \
    --dsmil_p8_weight ./ckpt/dsmil8_epoch_15_loss_0.3595_bacc_0.8958.pth \
    --dsmil_p16_weight ./ckpt/dsmil16_epoch_8_loss_0.6246_bacc_0.9583.pth \
    --num_worker 64 \
    --remove_label \
    --save_result_df ./push_code/result_df/result_lc_dsmil_all_patch.csv \
    --infer_p8 \
    --infer_p16 \
    --infer_dsmil \
    --dsmil_p8 0.5 \
    --dsmil_p16 0.5




# 15. infer transmil  lc
python ./push_code/src/inference_plot.py \
    --base_dir . \
    --test_csv ./push_code/out_csv/OC_lc.csv \
    --p8_feature_dir ./lc_40x_ViT_p8_448_0.5 \
    --p16_feature_dir ./lc_40x_ViT_p16_448_0.5 \
    --transmil_p8_weight /mnt/raid/zanzhuheng/working/NCOC/ckpt/transmil8_epoch_20_loss_0.5760_bacc_0.8854.pth \
    --transmil_p16_weight /mnt/raid/zanzhuheng/working/NCOC/ckpt/transmil16_epoch_20_loss_0.5213_bacc_0.9167.pth \
    --num_worker 64 \
    --remove_label \
    --save_result_df ./push_code/result_df/result_lc_transmil_all_patch.csv \
    --infer_p8 \
    --infer_p16 \
    --infer_transmil \
    --transmil_p8 0.5 \
    --transmil_p16 0.5



# 17. infer transmil  202
python ./push_code/src/inference_plot.py \
    --base_dir . \
    --test_csv ./push_code/out_csv/OC_202.csv \
    --p8_feature_dir ./202_40x_ViT_p8_448_0.5 \
    --p16_feature_dir ./202_40x_ViT_p16_448_0.5 \
    --transmil_p8_weight /mnt/raid/zanzhuheng/working/NCOC/ckpt/transmil8_epoch_20_loss_0.5760_bacc_0.8854.pth \
    --transmil_p16_weight /mnt/raid/zanzhuheng/working/NCOC/ckpt/transmil16_epoch_20_loss_0.5213_bacc_0.9167.pth \
    --num_worker 64 \
    --remove_label \
    --save_result_df ./push_code/result_df/result_202_transmil_all_patch.csv \
    --infer_p8 \
    --infer_p16 \
    --infer_transmil \
    --transmil_p8 0.5 \
    --transmil_p16 0.5







# infer all
python ./push_code/src/inference_plot.py \
    --base_dir . \
    --test_csv ./push_code/out_csv/OC_lc.csv \
    --p8_feature_dir ./lc_40x_ViT_p8_448_0.5 \
    --p16_feature_dir ./lc_40x_ViT_p16_448_0.5 \
    --abmil_p8_weight ./ckpt/weighted/abmil8_epoch_7_loss_0.2347_bacc_0.9583.pth \
    --abmil_p16_weight ./ckpt/weighted/abmil16_epoch_7_loss_0.2939_bacc_0.9375.pth \
    --dsmil_p8_weight ./ckpt/dsmil8_epoch_15_loss_0.3595_bacc_0.8958.pth \
    --dsmil_p16_weight ./ckpt/dsmil16_epoch_8_loss_0.6246_bacc_0.9583.pth \
    --num_worker 64 \
    --remove_label \
    --save_result_df ./push_code/result_df/result_lc_test.csv \
    --infer_p8 \
    --infer_p16 \
    --infer_abmil \
    --infer_dsmil \
    --abmil_p8 0.25 \
    --abmil_p16 0.25 \
    --dsmil_p8 0.25 \
    --dsmil_p16 0.25


python ./push_code/src/inference_plot.py \
    --base_dir . \
    --test_csv ./push_code/out_csv/OC_202.csv \
    --p8_feature_dir ./202_40x_ViT_p8_448_0.5 \
    --p16_feature_dir ./202_40x_ViT_p16_448_0.5 \
    --abmil_p8_weight ./ckpt/weighted/abmil8_epoch_7_loss_0.2347_bacc_0.9583.pth \
    --abmil_p16_weight ./ckpt/weighted/abmil16_epoch_7_loss_0.2939_bacc_0.9375.pth \
    --dsmil_p8_weight ./ckpt/dsmil8_epoch_15_loss_0.3595_bacc_0.8958.pth \
    --dsmil_p16_weight ./ckpt/dsmil16_epoch_8_loss_0.6246_bacc_0.9583.pth \
    --num_worker 64 \
    --remove_label \
    --save_result_df ./push_code/result_df/result_202_test.csv \
    --infer_p8 \
    --infer_p16 \
    --infer_abmil \
    --infer_dsmil \
    --abmil_p8 0.25 \
    --abmil_p16 0.25 \
    --dsmil_p8 0.25 \
    --dsmil_p16 0.25