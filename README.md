## Viusal Effects
### Displayments of the Motion4D Models
Just open the file index.html to view the visual effects and introduction.



## Setup
### Requirements

```shell
# create virtual environment
conda create -n motiondirector python=3.8
conda activate motiondirector
# install packages
pip install -r requirements.txt
```

### Weights of Foundation Models
```shell
git lfs install
## You can choose the ModelScopeT2V or ZeroScope, etc., as the foundation model.
## ZeroScope
git clone https://huggingface.co/cerspense/zeroscope_v2_576w ./models/zeroscope_v2_576w/
## ModelScopeT2V
git clone https://huggingface.co/damo-vilab/text-to-video-ms-1.7b ./models/model_scope/
```
### Weights of trained MotionDirector <a name="download_weights"></a>
```shell
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/ruizhaocv/MotionDirector_weights ./outputs

# More and better trained MotionDirector are released at a new repo:
git clone https://huggingface.co/ruizhaocv/MotionDirector ./outputs
# The usage is slightly different, which will be updated later.
```

## Usage
### Training

#### Train MotionDirector on multiple videos:
```bash
python MotionDirector_train.py --config ./configs/config_multi_videos.yaml
```
#### Train MotionDirector on a single video:
```bash
python MotionDirector_train.py --config ./configs/config_single_video.yaml
```

Note:  
- Before running the above command, 
make sure you replace the path to foundational model weights and training data with your own in the config files `config_multi_videos.yaml` or `config_single_video.yaml`.
- Generally, training on multiple 16-frame videos usually takes `300~500` steps, about `9~16` minutes using one A5000 GPU. Training on a single video takes `50~150` steps, about `1.5~4.5` minutes using one A5000 GPU. The required VRAM for training is around `14GB`.
- Reduce `n_sample_frames` if your GPU memory is limited.
- Reduce the learning rate and increase the training steps for better performance.


### Inference
```bash
python MotionDirector_inference.py --model /path/to/the/foundation/model  --prompt "Your prompt" --checkpoint_folder /path/to/the/trained/MotionDirector --checkpoint_index 300 --noise_prior 0.
```
Note: 
- Replace `/path/to/the/foundation/model` with your own path to the foundation model, like ZeroScope.
- The value of `checkpoint_index` means the checkpoint saved at which the training step is selected.
- The value of `noise_prior` indicates how much the inversion noise of the reference video affects the generation. 
We recommend setting it to `0` for MotionDirector trained on multiple videos to achieve the highest diverse generation, while setting it to `0.1~0.5` for MotionDirector trained on a single video for faster convergence and better alignment with the reference video.


## Inference with pre-trained MotionDirector
All available weights are at official [Huggingface Repo](https://huggingface.co/ruizhaocv/MotionDirector_weights).
Run the [download command](#download_weights), the weights will be downloaded to the folder `outputs`, then run the following inference command to generate videos.

### MotionDirector trained on multiple videos:
```bash
python MotionDirector_inference.py --model /path/to/the/ZeroScope  --prompt "A person is riding a bicycle past the Eiffel Tower." --checkpoint_folder ./outputs/train/riding_bicycle/ --checkpoint_index 300 --noise_prior 0. --seed 7192280
```
Note:  
- Replace `/path/to/the/ZeroScope` with your own path to the foundation model, i.e. the ZeroScope.
- Change the `prompt` to generate different videos. 
- The `seed` is set to a random value by default. Set it to a specific value will obtain certain results, as provided in the table below.



### MotionDirector trained on a single video:
16 frames:
```bash
python MotionDirector_inference.py --model /path/to/the/ZeroScope  --prompt "A tank is running on the moon." --checkpoint_folder ./outputs/train/car_16/ --checkpoint_index 150 --noise_prior 0.5 --seed 8551187
```


24 frames:
```bash
python MotionDirector_inference.py --model /path/to/the/ZeroScope  --prompt "A truck is running past the Arc de Triomphe." --checkpoint_folder ./outputs/train/car_24/ --checkpoint_index 150 --noise_prior 0.5 --width 576 --height 320 --num-frames 24 --seed 34543
```


## MotionDirector for Sports <a name="MotionDirector_for_Sports"></a>

```bash
python MotionDirector_inference.py --model /path/to/the/ZeroScope  --prompt "A panda is lifting weights in a garden." --checkpoint_folder ./outputs/train/lifting_weights/ --checkpoint_index 300 --noise_prior 0. --seed 9365597
```



More sports, to be continued ...






## MotionDirector with Customized Appearance <a name="MotionDirector_with_Customized_Appearance"></a>
### Train
Train the spatial path with reference images.
```bash
python MotionDirector_train.py --config ./configs/config_multi_images.yaml
```
Then train the temporal path to learn the motions in reference videos.
```bash
python MotionDirector_train.py --config ./configs/config_multi_videos.yaml
```

### Inference
Inference with spatial path learned from reference images and temporal path learned form reference videos.
```bash
python MotionDirector_inference_multi.py --model /path/to/the/foundation/model  --prompt "Your prompt" --spatial_path_folder /path/to/the/trained/MotionDirector/spatial/lora/ --temporal_path_folder /path/to/the/trained/MotionDirector/temporal/lora/ --noise_prior 0.
```
### Example
Download the pre-trained weights.
```bash
git clone https://huggingface.co/ruizhaocv/MotionDirector ./outputs
```
Run the following command.
```bash
python MotionDirector_inference_multi.py --model /path/to/the/ZeroScope  --prompt "A Terracotta Warrior is riding a horse through an ancient battlefield." --spatial_path_folder ./outputs/train/customized_appearance/terracotta_warrior/checkpoint-default/spatial/lora --temporal_path_folder ./outputs/train/riding_horse/checkpoint-default/temporal/lora/ --noise_prior 0. --seed 1455028
```
Results are shown in the [table](#customize-both-appearance-and-motion-).

## More results

If you have a more impressive MotionDirector or generated videos, please feel free to open an issue and share them with us. We would greatly appreciate it.
Improvements to the code are also highly welcome.

Please refer to [Project Page](https://showlab.github.io/MotionDirector) for more results.



## Shoutouts

- This code builds on [diffusers](https://github.com/huggingface/diffusers), [Tune-a-video](https://github.com/showlab/Tune-A-Video) and [Text-To-Video-Finetuning](https://github.com/ExponentialML/Text-To-Video-Finetuning). Thanks for open-sourcing!
- Thanks to [camenduru](https://twitter.com/camenduru) for the [colab demo](https://github.com/camenduru/MotionDirector-colab).
- Thanks to [yhyu13](https://github.com/yhyu13) for the [Huggingface Repo](https://huggingface.co/Yhyu13/MotionDirector_LoRA).
- We would like to thank [AK(@_akhaliq)](https://twitter.com/_akhaliq?lang=en) and huggingface team for the help of setting up oneline gradio demo.
- Thanks to [MagicAnimate](https://github.com/magic-research/magic-animate/) for the gradio demo template.
- Thanks to [deepbeepmeep](https://github.com/deepbeepmeep), and [XiaominLi](https://github.com/XiaominLi1997) for improving the code.





## Install
```bash
# python 3.10 cuda 11.8 
conda create -n dg4d python=3.10 -y && conda activate dg4d
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.23 --no-deps --index-url https://download.pytorch.org/whl/cu118

# other dependencies
pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# simple-knn
pip install ./simple-knn

# for mesh extraction
pip install git+https://github.com/NVlabs/nvdiffrast/
```

To use pretrained LGM:

```bash
# for LGM
mkdir pretrained && cd pretrained
wget https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors
cd ..
```



## Image-to-4D
##### (Optional) Preprocess input image
```bash
python scripts/process.py data/anya.png
```
##### Step 1: Generate driving videos
```bash
python scripts/gen_vid.py --path data/anya_rgba.png --seed 42 --bg white
```
##### Step 2: static generation
Static generation with [LGM](https://github.com/3DTopia/LGM):
```bash
python lgm/infer.py big --test_path data/anya_rgba.png
```
Optionally, we support static generation with [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian):
```bash
python dg.py --config configs/dg.yaml input=data/anya_rgba.png
```
See `configs/dghd.yaml` for high-quality DreamGaussian training configurations.

##### Step 3: dynamic generation
```bash
# load static 3D from LGM
python main_4d.py --config configs/4d.yaml input=data/anya_rgba.png

# (Optional) to load static 3D from DreamGaussian, add `radius=2`
python main_4d.py --config configs/4d.yaml input=data/anya_rgba.png radius=2

# (Optional) to turn on viser GUI, add `gui=True`, e.g.:
python main_4d.py --config configs/4d.yaml input=data/anya_rgba.png gui=True
```
See `configs/4d_low.yaml` and `configs/4d_demo.yaml` for more memory-friendly and faster optimization configurations.

##### (Optional) Step 4: mesh refinment
```bash
# export mesh after temporal optimization by adding `mesh_format=obj`
python main_4d.py --config configs/4d.yaml input=data/anya_rgba.png mesh_format=obj

# mesh refinement
python main2_4d.py --config configs/refine.yaml input=data/anya_rgba.png

# (Optional) to load static 3D from DreamGaussian, add `radius=2`
python main2_4d.py --config configs/refine.yaml input=data/anya_rgba.png radius=2
```

## Video-to-4D
##### Prepare Data
Download [Consistent4D data](https://consistent4d.github.io/) to `data/CONSISTENT4D_DATA`. `python scripts/add_bg_to_gt.py` will add white background to ground-truth novel views.

##### Step 1: static generation
```bash
python lgm/infer.py big --test_path data/CONSISTENT4D_DATA/in-the-wild/blooming_rose/0.png

# (Optional) static 3D generation with DG
python dg.py --config configs/dg.yaml input=data/CONSISTENT4D_DATA/in-the-wild/blooming_rose/0.png
```

##### Step 2: dynamic generation
```bash
python main_4d.py --config configs/4d_c4d.yaml input=data/CONSISTENT4D_DATA/in-the-wild/blooming_rose

# (Optional) to load static 3D from DG, add `radius=2`
python main_4d.py --config configs/4d_c4d.yaml input=data/CONSISTENT4D_DATA/in-the-wild/blooming_rose radius=2
```



## Tips
 
- Black video after running `gen_vid.py`.
    - Make sure pytorch version is >=2.0 



## Acknowledgement
## Note
This project contains two stage of text to 4D generation. The first stage T2V is modified from motionderector. The second stage is intensively reconstructed from Dreamgaussian4D. We appreaciate it for their great effort. 
This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!
* [4DGaussians](https://github.com/hustvl/4DGaussians)
* [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian)
* [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
* [threestudio](https://github.com/threestudio-project/threestudio)
* [nvdiffrast](https://github.com/NVlabs/nvdiffrast)









