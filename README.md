# IONet
## Overview of IONet
 ![image](https://github.com/user-attachments/assets/41aec451-e10f-4729-ae8e-c7cd159ac585)



Since we ensemble two MIL models [ABMIL](https://arxiv.org/abs/1802.04712) and [DSMIL](https://arxiv.org/abs/2011.08939) and extractor used [DINO-ViT](https://arxiv.org/abs/2104.14294).

if you have any problems please contact ***wacto1998@gamil.com*** <br>
DINO-ViT pretrained weight can be found in [this link](https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights).

![image](https://github.com/user-attachments/assets/2c860d7b-9e10-4d63-9154-a9afbcb61f6a)
![image](https://github.com/user-attachments/assets/824bbd08-6f04-4154-bbe4-8c425163f3c6)

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

to be update
