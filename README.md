# IONet

## Overview of IONet
 ![image](https://github.com/user-attachments/assets/41aec451-e10f-4729-ae8e-c7cd159ac585)

<p style="font-style: italic; font-family: 'Times New Roman';font-size:16px">
Figure 1 | IONet's methodology begins by extracting 448x448 patches and selecting regions of interest (ROIs) with relevant information, which are downsampled to 224x224. These ROIs are then processed by a pre-trained <a href="https://arxiv.org/abs/2104.14294">DINO-ViT</a> model, fine-tuned on a large pathology dataset, segmenting the patches into detailed mini-patches at cellular and tissue scales. The extracted features are passed through a multiscale feature-learning module using a Transformer architecture, generating a robust visual representation of the ROI. we ensemble <a href="https://arxiv.org/abs/1802.04712">ABMIL</a> and <a href="https://arxiv.org/abs/2011.08939">DSMIL</a> models to enhance performance and robustness in our final predictions. Further details can be found in the Methodology section.
</p>

if you have any problems please contact ***wacto1998@gamil.com*** <br>
DINO-ViT pretrained weight can be found in [this link](https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights).

![image](https://github.com/user-attachments/assets/2c860d7b-9e10-4d63-9154-a9afbcb61f6a)

<p style="font-style: italic; font-family: 'Times New Roman'; font-size: 16px">
Figure 2 | The visualization for ovarian cancer subtyping tasks is presented as follows:<br>
a. The sequence of four images, from left to right, includes:<br>
   1. A representative slide from the HGSC category, selected to generate corresponding visualization results.<br>
   2. A whole-slide attention heatmap generated for each slide by calculating the attention scores for the HGSC class.<br>
   3. Visualization merging representative slide images with their corresponding heatmap for the HGSC class.<br>
   4. Patches with high attention scores from the representative slide effectively reflect the typical morphological features of the HGSC class.<br>
b. The visualization results for the CCOC subtype in ovarian cancer are analogous to the four images in the first row, from left to right.<br>
c. The visualization results for the LGSC subtype in ovarian cancer are analogous to the four images in the first row, from left to right.<br>
d. The visualization results for the ECOC subtype in ovarian cancer are analogous to the four images in the first row, from left to right.
</p>




## enviroment set-up
### hardward 
We use RTX 4090 24G GPU for training and inference.

### software
**System**: Ubuntu 22.04
**Python and DeepLearning framwork** : python 3.10 and pytorch 2.1.2+cu118

we provide both conda *environment.yaml* and *requirments.txt* for readers to reproduce our results.⬇

run following command to create conda environment⬇️

```bash
# create conda environment
conda create -n ionet python=3.10 
conda activate ionet

# install dependencies
pip install -r requirments.txt
# or
conda env create -f environment.yaml

# install pyvips
unzip ./install/pyvips.zip 

sudo apt-get update
sudo apt-get install libvips-dev -y --no-install-recommends --download-only -o dir::cache='./'

mkdir ./libvips
sudo mv ./archives/* ./libvips    
rm -rf ./archives
ls ./libvips

yes | sudo dpkg -i .install/pyvips/libvips/*.deb

# pip
pip install pyvips
pip wheel pyvips
mkdir pyvips
mv *.whl ./pyvips
```
## Data pre-process
run
`sh ./shell.preprocess.sh`

You may modify some paths before run preprocess.sh

Behaviors: <br>

0. make datafram for train
1. make patches
2. extract features

## Training
run
`sh ./shells/train.sh`

You may modify some paths before run preprocess.sh

Behaviors: <br>

0. train abmil patch size 16
1. train abmil patch size 8
2. train dsmil patch size 8
3. train dsmil patch size 16

## Inference
run 
`./shells/infer.sh`
You may modify some paths before run preprocess.sh

Behaviors: <br>

0. infer models with average weights

## Generate heatmap
run
`sh ./shells/heatmap.sh`
You may modify some paths before run preprocess.sh

Behaviors: <br>

0. generate heatmap


to be update


to be update
