python ./source/make_heatmap.py \
    --base_dir . \
    --test_csv ./out_csv/your.csv \
    --p8_feature_dir ./ViT_p8_448_0.5 \
    --p16_feature_dir ./ViT_p16_448_0.5 \
    --abmil_p8_weight ./ckpt/weighted/abmil8.pth \
    --abmil_p16_weight ./ckpt/weighted/abmil16.pth \
    --dsmil_p8_weight ./ckpt/dsmil8.pth \
    --dsmil_p16_weight ./ckpt/dsmil16.pth \
    --num_worker 64 \
    --heatmap_save_dir ./out_heatmap \
    --patch_dir ./40x_448_0.5 \

