python ./push_code/heatmap/make_heatmap.py \
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
    --heatmap_save_dir ./push_code/heatmap/out_heatmap_202 \
    --patch_dir ./202_40x_448_0.5 \



python ./push_code/heatmap/make_heatmap.py \
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
    --heatmap_save_dir ./push_code/heatmap/out_heatmap_lc \
    --patch_dir ./lc_40x_448_0.5



python ./push_code/heatmap/make_heatmap.py \
    --base_dir . \
    --test_csv ./push_code/out_csv/OC_sc.csv \
    --p8_feature_dir ./sc_40x_ViT_p8_448_0.5 \
    --p16_feature_dir ./sc_40x_ViT_p16_448_0.5 \
    --abmil_p8_weight ./ckpt/weighted/abmil8_epoch_7_loss_0.2347_bacc_0.9583.pth \
    --abmil_p16_weight ./ckpt/weighted/abmil16_epoch_7_loss_0.2939_bacc_0.9375.pth \
    --dsmil_p8_weight ./ckpt/dsmil8_epoch_15_loss_0.3595_bacc_0.8958.pth \
    --dsmil_p16_weight ./ckpt/dsmil16_epoch_8_loss_0.6246_bacc_0.9583.pth \
    --num_worker 64 \
    --remove_label \
    --heatmap_save_dir ./push_code/heatmap/out_heatmap_sc \
    --patch_dir ./sc_40x_448_0.5

