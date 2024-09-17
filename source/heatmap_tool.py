import os
import pdb
import pyvips
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.ndimage import gaussian_filter
import cv2

import os
import pyvips
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2

def draw_heatmap(save_dir, slide_id, patches_dir, attention_matrix_dic, svs_file, overlap=False):
    """
    Draw heatmap for each WSI 

    Args:
        save_dir (str): directory to save heatmap and overlay images
        slide_id (str): slide id
        patches_dir (str): directory for patches
        attention_matrix_dic (dict): dictionary containing attention matrix for each method
        svs_file (str): path to the original SVS file
        overlap (bool): whether to overlap a heatmap on the WSI file
    """
    os.makedirs(save_dir, exist_ok=True)

    patch_size = 16
    patch_pos = []
    folder_path = os.path.join(patches_dir, slide_id)
    print("Processing Slide, ID: " + slide_id)

    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            image_id = os.path.splitext(file_name)[0]
            x, y = image_id.split('_')[-1].split('-')
            patch_pos.append((int(x) * patch_size, int(y) * patch_size))
    patch_pos = np.array(patch_pos)
    max_x, max_y = np.amax(patch_pos, axis=0)

    all_attention_matrix = None
    for method in attention_matrix_dic.keys():
        attention_matrix = attention_matrix_dic[method]
        attention_matrix = (attention_matrix - attention_matrix.min()) / (attention_matrix.max() - attention_matrix.min())
        if all_attention_matrix is None:
            all_attention_matrix = attention_matrix
        else:
            all_attention_matrix += attention_matrix
    all_attention_matrix = (all_attention_matrix - all_attention_matrix.min()) / (all_attention_matrix.max() - all_attention_matrix.min())
    attention_matrix_dic['all'] = all_attention_matrix

    
    original_image = pyvips.Image.new_from_file(svs_file[0], access='sequential')
    original_width = original_image.width
    original_height = original_image.height

    for method in attention_matrix_dic.keys():
        attention_matrix = attention_matrix_dic[method]
        if method != 'all':
            attention_matrix = (attention_matrix - attention_matrix.min()) / (attention_matrix.max() - attention_matrix.min())
        os.makedirs(os.path.join(save_dir, method), exist_ok=True)
        cmap = plt.cm.get_cmap('rainbow')
        heatmap = np.ones((max_y + 5, max_x + 5, 3), dtype=np.uint8) * 255  

        for patch_coord, patch_score in zip(patch_pos, attention_matrix):
            x, y = patch_coord[0], patch_coord[1]
            h = patch_size
            color = cmap(1.0 - patch_score)
            color_rgb = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            heatmap[y:y + h, x:x + h] = color_rgb

        # Convert heatmap to a format that matplotlib can save as PDF
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        heatmap_pdf_path = os.path.join(save_dir, method, slide_id + '_heatmap.pdf')
        plt.savefig(heatmap_pdf_path, format='pdf', bbox_inches='tight', pad_inches=0)
        plt.close()

        if overlap:
            heatmap_height, heatmap_width, _ = heatmap.shape

            # Calculate scale factors
            scale_x = heatmap_width / original_width
            scale_y = heatmap_height / original_height
            scale = min(scale_x, scale_y)  # Use the smaller scale to maintain aspect ratio

            # Resize the original image using the calculated scale factor
            resized_svs_image = original_image.resize(scale, kernel='linear')

            # Convert the resized SVS image to an array
            svs_image_array = np.ndarray(buffer=resized_svs_image.write_to_memory(), dtype=np.uint8, shape=(resized_svs_image.height, resized_svs_image.width, 3))

            # Ensure heatmap and SVS image have the same dimensions and data type
            if heatmap.shape[0] != svs_image_array.shape[0] or heatmap.shape[1] != svs_image_array.shape[1]:
                heatmap = cv2.resize(heatmap, (svs_image_array.shape[1], svs_image_array.shape[0]))

            alpha = 0.5  
            overlay_image = cv2.addWeighted(svs_image_array, 1 - alpha, heatmap, alpha, 0)

            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')

            overlay_pdf_path = os.path.join(save_dir, method, slide_id + '_overlay.pdf')
            plt.savefig(overlay_pdf_path, format='pdf', bbox_inches='tight', pad_inches=0)
            plt.close()



def to_percentiles(scores):
    """Convert attention scores to percentiles"""
    scores = rankdata(scores, 'average') / len(scores) * 100   
    return scores


def A_fliter(score,thr):
    """fliter score by threshold"""
    if thr is None:
        score[score < thr] = 0
    return score



