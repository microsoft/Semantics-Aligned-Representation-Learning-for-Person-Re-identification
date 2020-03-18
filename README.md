## Introduction
This repository holds the codes and methods for the following AAAI-2020 paper:
- [Semantics-Aligned Representation Learning for Person Re-identification](https://arxiv.org/pdf/1905.13143.pdf)

Person re-identification (reID) aims to match person images to retrieve the ones with the same identity. This is a challenging task, as the images to be matched are generally semantically misaligned due to the diversity of human poses and capture viewpoints, incompleteness of the visible bodies (due to occlusion), etc. 

We propose a framework that drives the reID network to learn semantics-aligned feature representation through delicate supervision designs. Specifically, we build a Semantics Aligning Network (SAN) which consists of a base network as encoder (SA-Enc) for re-ID, and a decoder (SA-Dec) for reconstructing/regressing the densely semantics aligned full texture image. We jointly train the SAN under the supervisions of person re-identification and aligned texture generation. Moreover, at the decoder, besides the reconstruction loss, we add Triplet ReID constraints over the feature maps as the perceptual losses. The decoder is discarded in the inference and thus our scheme is computationally efficient. Our design significantly outperforms the performance of baseline and achieve the state-of-the-art performance. 

![image](https://github.com/microsoft/Semantics-Aligned-Representation-Learning-for-Person-Re-identification/blob/master/pipeline.png)

Figure 1: Illustration of the proposed Semantics Aligning Network (SAN). It consists of a base network as encoder (SA-Enc) and a decoder sub-network (SA-Dec). The reID feature vector is obtained by average pooling the feature map of the SA-Enc, followed by the reID losses. To encourage the encoder learning semantically aligned features, the SA-Dec is followed which regresses the densely semantically aligned full texture image with the pseudo groundtruth supervision. In inference, the SA-Dec is discarded.

## Installation
1. Git clone this repo.
2. Install dependencies by `pip install -r requirements.txt` (if necessary).
3. To install the cython-based evaluation toolbox, `cd` to `torchreid/eval_cylib` and do `make`. As a result, `eval_metrics_cy.so` is generated under the same folder. Run `python test_cython.py` to test if the toolbox is installed successfully. (credit to [luzai](https://github.com/luzai))

## Synthesized Paired-Image-Texture dataset (PIT dataset)

To train the SAN-PG, we synthesize a Paired-Image-Texture dataset (PIT dataset), based on SURREAL dataset (https://www.di.ens.fr/willow/research/surreal/), for the purpose of providing the image pairs, i.e., the person image and its texture image. The texture image stores the RGB texture of the full person 3D surface. In particular, we use 929 raster-scanned texture maps provided by the SURREAL dataset to generate the image pairs. On SURREAL, all faces in the texture image are replaced by an average face of either man or woman. We generate 9,290 different meshes of diverse poses/shapes/viewpoints. For each texture map, we assign 10 different meshes and render these 3D meshes with the texture image. Then we obtain in total 9,290 different synthesized (person image, texture image) pairs. To simulate real-world scenes, the background images for rendering are randomly sampled from COCO dataset (http://cocodataset.org/#home). Each synthetic person image is centered on a person with resolution 256x128. The resolution of the texture images is 256x256. The PIT dataset can be download from [here](https://drive.google.com/file/d/1-ndIFhppMG_zjHCRfrnWRvbRQZObw2tT/view?usp=sharing)

![image](https://github.com/microsoft/Semantics-Aligned-Representation-Learning-for-Person-Re-identification/blob/master/PIT_dataset.png)


Figure 2: Examples of texture images (first row) and the corresponding synthesized person images with different poses, viewpoints, and backgrounds (second row). A texture image represents the full texture of the 3D human surface in a surface-based canonical coordinate system (UV space). Each position (u,v) corresponds to a unique semantic identity. For person images of different persons/poses/viewpoints (in the second row), their corresponding texture images are densely semantically aligned.

## ReID Dataset Preparation
Image-reid datasets (here we use CUHK03 dataset as an example for description):
- [CUHK03](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf) (`cuhk03`)
The keys to use these datasets are enclosed in the parentheses. See [torchreid/datasets/\_\_init__.py](torchreid/datasets/__init__.py) for details. The data managers of image reid are implemented in [torchreid/data_manager.py](torchreid/data_manager.py).
1. Create a folder named `cuhk03/` under `/YOUR_DATASET_PATH/`.
2. Download dataset to `data/cuhk03/` from http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html and extract `cuhk03_release.zip`, so you will have `data/cuhk03/cuhk03_release`.
3. Download new split from [person-re-ranking](https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03). What you need are `cuhk03_new_protocol_config_detected.mat` and `cuhk03_new_protocol_config_labeled.mat`. Put these two mat files under `data/cuhk03`. Finally, the data structure would look like
```
cuhk03/
    cuhk03_release/
    cuhk03_new_protocol_config_detected.mat
    cuhk03_new_protocol_config_labeled.mat
    ...
```
4. Use `-d cuhk03` when running the training code. In default mode, we use new split (767/700). In addition, here we use both `labeled` modes. Please specify `--cuhk03-labeled` to train and test on `labeled` images.


## Pseudo Groundtruth Texture Images Generation

We train a network for the purpose of generating pseudo groundtruth texture images for any given input person image. For simplicity, we reuse a simplified SAN (i.e., SAN-PG) which consists of the SA-Enc and SA-Dec, but with only the reconstruction loss. We train the SAN-PG with our synthesized PIT dataset. The SAN-PG model is then used to generate pseudo groundtruth texture image for reID dataset.

Here we provide the pre-trained weight for [SAN-PG](https://drive.google.com/file/d/14awPVhJA5yT9j7nZb7nVAmC3J6ELxNGD/view?usp=sharing) and the corresponding pseudo texture images generation scripts `generate_texture.py`, you can synthesize your own texture images by running:
```bash
python generate_texture.py -m /DOWNLOADED_SAN-PG_WEIGHTS -i example_results/input -o example_results/texture
```

For convenience, we also provide our pre-synthesized pseudo groundtruth texture images for CUHK03 (Labeled), that is [texture_cuhk03_labeled](https://drive.google.com/file/d/19-9WdlbqjD4n2usV-D2zyyzeUfjcXxlv/view?usp=sharing) 

- Extract our pre-synthesized pseudo groundtruth texture images to /YOUR_DATASET_PATH/cuhk03/
- Finally, the CUHK03 ReID data structure would look like
```
cuhk03/
    cuhk03_release/
    cuhk03_new_protocol_config_detected.mat
    cuhk03_new_protocol_config_labeled.mat
    texture_cuhk03_labeled
    ...
```


## Training and evaluation

```bash
python main.py \
--root DATASET_PATH \
-s cuhk03 \
-t cuhk03 
--height 256 \
--width 128 \
--optim amsgrad \
--label-smooth \
--lr 8e-04 \
--max-epoch 300 \
--stepsize 40 80 120 160 200 240 280 \
--train-batch-size 64 \
--test-batch-size 100
-a resnet50_fc512 \
--save-dir SAVE_PATH \
--gpu-devices 0 \
--train-sampler RandomIdentitySampler \
--warm-up-epoch 20 \
--cuhk03-labeled \
--eval-freq 80
```

## Reference
If you find our papers and repo useful, please cite our paper. Thanks!

```
@article{jin2020semantics,
  title={Semantics-aligned representation learning for person re-identification},
  author={Jin, Xin and Lan, Cuiling and Zeng, Wenjun and Wei, Guoqiang and Chen, Zhibo},
  journal={AAAI},
  year={2020}
}
```
Microsoft Open Source Code of Conduct: https://opensource.microsoft.com/codeofconduct


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.