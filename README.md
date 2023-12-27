# Conducting Comprehensive Training and Validation Tests with Maritime Dataset and Fine-Tuned Hyperparameters for [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

Explore an all-in-one Colab notebook to reproduce my results using the pre-trained model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gn4E-sGde88apPNuxCiONjOaJJ1FrIU7?usp=sharing)

If you wish to conduct your experiments using maritime data with fine-tuned hyperparameters, please continue reading.

## Overview of CycleGAN

CycleGAN, or Cycle-Consistent Generative Adversarial Network, is a deep learning model designed for image-to-image translation. Developed in 2017 by researchers at the University of California, Berkeley, CycleGAN's primary objective is to learn mappings between two domains without the need for paired training data. The implementation can be found [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

<details>
  <summary>(Click to Expand) <strong>Key Components and Concepts Associated with CycleGAN:</strong></summary>

1. **Generative Adversarial Networks (GANs):** CycleGAN builds upon the foundation of GANs. This architecture consists of a generator and a discriminator network trained simultaneously through adversarial training. The generator creates realistic-looking data, while the discriminator distinguishes between real and generated data.

2. **Cycle-Consistency:** A distinctive feature of CycleGAN is its emphasis on cycle-consistency. In image translation tasks, cycle-consistency ensures that if an image from domain A is translated to domain B and then back to domain A, it should resemble the original image. This constraint is enforced during training to improve the quality of generated images.

3. **Unpaired Image Translation:** Unlike many other image translation models requiring paired training data, CycleGAN can learn mappings between domains without such pairs. This makes it particularly useful in scenarios where obtaining paired data is challenging or impractical.

4. **Loss Functions:** CycleGAN uses multiple loss functions for effective model training. The adversarial loss encourages the generator to create realistic images, and the cycle-consistency loss enforces consistency between the original and reconstructed images after translation.

5. **Applications:** CycleGAN has been successfully applied to various image translation tasks, such as turning photographs into paintings, converting satellite images to maps, transforming horses into zebras, and more. Its versatility and ability to handle unpaired data make it suitable for a wide range of domains.

</details>

## Overview of Datasets Used in This Experiment

In this experiment, we leveraged two public datasets within the Maritime domain:

1. **Domain A (Photorealistic Data):** [The Split Port Ship Classification Dataset (SPSCD)](https://labs.pfst.hr/maritime-dataset/)
2. **Domain B (Simulated):** [A High Resolution Simulation Dataset for Ship Detection with Precise Annotations](https://research.abo.fi/en/datasets/simuships-a-high-resolution-simulation-dataset-for-ship-detection)

### Experiment Objective with Cycle-GAN

The primary objective of employing Cycle-GAN in this experiment is to train a model with a generator_B2A. This generator is designed to apply the image style of Domain A (Photorealistic Data) onto Domain B (Simulated Data). The goals are twofold:

1. **Discriminator A Recognition:**
   The primary task of discriminator A is to exert its utmost effort in distinguishing between the authentic photorealistic images and the newly generated photorealistic images produced by generator_A2B.

2. **Generator_A2B Cyclical Image Generation:**
   Generator_A2B is tasked with creating new photorealistic images that closely emulate the styles inherent in the original photorealistic images. The objective is to deceive discriminator A to the maximum extent possible, making it challenging for the discriminator to differentiate between genuine and generated photorealistic images.

![Experimental Overview](./imgs/cycleGAN.png)

## Overview of Environment Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/lunapapa-finland/EDISS-CyclyGAN.git
   ```

2. **Install the required packages. You can use either pip or conda for environment control:**

   Using pip:

   ```bash
   pip install -r requirements.txt
   ```

   Or using conda:

   ```bash
   conda install --file requirements.txt
   ```

3. **Download a tar file in the root of the cloned repository:**

   ```bash
   gdown 1z1SQPF40atK_Tfxw0gjsAW1C8lBKjPnB --output OBS.tar
   ```

   This command uses `gdown` to download a tar file containing necessary extra setups as well as small datasets.

4. **Extract the contents of the tar file:**

   ```bash
   tar -xvf OBS.tar
   ```

5. **Remove the downloaded tar file:**

   ```bash
   rm OBS.tar
   ```

   This command removes the "OBS.tar" file, freeing up disk space after its contents have been extracted.

6. **Navigate to the OBS folder and grant execution permissions to the following bash files:**

   ```bash
   cd OBS
   chmod +x org.sh create_checkpoints_structure.sh create_datasets_structure.sh
   ```
  
  This command enables the execution rights for three essential bash scripts: org.sh, create_checkpoints_structure.sh, and create_datasets_structure.sh. These scripts facilitate the automatic creation or organization of structures for datasets and checkpoints.

## Usage Details for Makefile Commands

In the subsequent sections dedicated to training and testing, the configuration of hyperparameters in the CycleGAN setup is intricate. We will employ not only Python commands but also utilize bash commands for file manipulation. To streamline workflow control and enhance maintenance, we will consistently leverage the `make` command.

To explore the available `make` commands, simply type the following in the command line:

```bash
make
```

This command will display all the make commands specified in the MakeFile. Subsequently, you can choose to execute specific commands or formulate your own set of commands following a similar pattern. It is crucial to note that you should always be operating within the `OBS directory`, not the root folder, as all optimizations and workflows are configured in the OBS folder

The output will be:

- `create_checkpoints_structure`: Create folder structure for preparation of pretrained checkpoint
- `create_datasets_structure`: Create folder structure for datasets
- `org_structure`: Copy datasets and checkpoints to the right places
- `test_SPSCDvsALSHD_random_Gresnet_DnLayer`: Testing Model with Generator_resnet9B and Discriminator nLayer
- `test_SPSCDvsALSHD_random_Gresnet_DPatchGAN`: Testing Model with Generator_resnet9B and Discriminator PatchGaN
- `test_SPSCDvsALSHD_random_Gunet_256_DnLayer`: Testing Model with Generator_unet256 and Discriminator nLayer
- `test_SPSCDvsALSHD_random_Gunet_256_DPatchGAN`: Testing Model with Generator_unet256 and Discriminator PatchGaN
- `test_SPSCDvsALSHD_selected_Gresnet_DnLayer`: Testing Model with Generator_resnet9B and Discriminator nLayer
- `test_SPSCDvsALSHD_selected_Gresnet_DPatchGAN`: Testing Model with Generator_resnet9B and Discriminator PatchGaN
- `test_SPSCDvsALSHD_selected_Gunet_256_DnLayer`: Testing Model with Generator_unet256 and Discriminator nLayer
- `test_SPSCDvsALSHD_selected_Gunet_256_DPatchGAN`: Testing Model with Generator_unet256 and Discriminator PatchGaN
- `train_SPSCDvsALSHD_random_Gresnet_DnLayer`: Training Model with Generator_resnet9B and Discriminator nLayer using random images
- `train_SPSCDvsALSHD_random_Gresnet_DPatchGAN`: Training Model with Generator_resnet9B and Discriminator PatchGaN using random images
- `train_SPSCDvsALSHD_random_Gunet_256_DnLayer`: Training Model with Generator_unet256 and Discriminator nLayer using random images
- `train_SPSCDvsALSHD_random_Gunet_256_DPatchGAN`: Training Model with Generator_unet256 and Discriminator PatchGaN using random images
- `train_SPSCDvsALSHD_selected_Gresnet_DnLayer`: Training Model with Generator_resnet9B and Discriminator nLayer using selected images
- `train_SPSCDvsALSHD_selected_Gresnet_DPatchGAN`: Training Model with Generator_resnet9B and Discriminator PatchGaN using selected images
- `train_SPSCDvsALSHD_selected_Gunet_256_DnLayer`: Training Model with Generator_unet256 and Discriminator nLayer using selected images
- `train_SPSCDvsALSHD_selected_Gunet_256_DPatchGAN`: Training Model with Generator_unet256 and Discriminator PatchGaN using selected images

## Training Steps Overview

For the purpose of illustration, we will perform a single training process. If you wish to train your own datasets, whether in the maritime domain or any other domain, simply follow the steps outlined below:

### Data Split Strategy

- **TrainA:** Select 80 images from Domain A with obvious objects (e.g., ship, shore, etc.).
- **TrainB:** Select 101 images from Domain B with obvious objects (e.g., ship, shore, etc.).
- **TestA:** Randomly select images from Domain A (No need to transfer images from photorealistic to simulated ones).
- **TestB:** Randomly select 200 images from Domain B (Same as the third experiment setup for cross-validation purposes).

In you have your own datasets or you want to change the samples in the datasets, simple run

```bash
# Command to create folder structure for datasets 
$ make create_datasets_structure
```

and you can copy your datasets to the correspondinng folder for further usage.

### Training Parameter Highlights

- **Generator:** Resnet 9 Block/Unet256
- **Discriminator:** PatchGAN/nLayer
- **Other Hyperparameters**

### Training Steps

```bash
# Command to copy checkpoints and datasets to corresponding folders in the root
$ make org_structure

# Command to start the training process for the model
$ make train_SPSCDvsALSHD_selected_Gresnet_DPatchGAN
```

It essentially runs the following command:

```bash
nohup python3 ../train.py \
    --dataroot ../datasets/SPSCDvsALSHD_selected \  # Specify the root directory for the training dataset
    --name SPSCDvsALSHD_selected_Gresnet_DPatchGAN \  # Set a name for the training run
    --model cycle_gan \  # Specify the type of model (CycleGAN in this case)
    --gpu_ids 0 \  # Specify the GPU to be used for training
    --gan_mode vanilla \  # Set the mode of GAN training to vanilla
    --pool_size 50 \  # Specify the size of the image buffer for discriminator updates
    --batch_size 1 \  # Set the size of each mini-batch used during training
    --checkpoints_dir ../checkpoints \  # Specify the directory where checkpoints will be saved
    --display_id -1 \  # Disable visual display during training
    --preprocess scale_width_and_crop \  # Specify the preprocessing method for input images
    --load_size 1920 \  # Set the size to load training images
    --crop_size 512 \  # Specify the size to crop training images
    --save_epoch_freq 5 \  # Specify the frequency (in epochs) to save checkpoints
    > ./checkpoints/SPSCDvsALSHD_selected_Gresnet_DPatchGAN/SPSCDvsALSHD_selected_Gresnet_DPatchGAN.log \  # Redirect standard output to a log file
    2>&1 &  # Redirect standard error to the same log file as standard output and run the command in the background
```

## Analysis of Training Logs

In this section, we will perform an analysis of our training models based on their corresponding logs. The training log contains 8 parameters that warrant analysis, and, in total, we will generate 9 graphs per log.

The structure of the training log is as follows:

```
(epoch: 153, iters: 100, time: 0.622, data: 0.313) D_A: 0.087 G_A: 3.190 cycle_A: 0.888 idt_A: 0.481 D_B: 0.011 G_B: 4.768 cycle_B: 1.115 idt_B: 0.387
```

### Parameters to Analyze

- **Discriminator Loss (D_A, D_B):** Represent the loss of discriminators D_A and D_B. Lower values indicate better performance.

- **Generator Loss (G_A, G_B):** Represent the loss of generators G_A and G_B. The generator aims to minimize this loss by generating realistic data.

- **Cycle Consistency Loss (cycle_A, cycle_B):** Associated with cycle consistency regularization, ensuring consistency in image translation.

- **Identity Loss (idt_A, idt_B):** Associated with identity mapping regularization, ensuring consistency in translated images.

\[ \text{Target\_Loss} = \min_{\text{epochs}} \left( \text{D\_B\_values} + \text{G\_B\_values} + \text{cycle\_B\_values} + \text{idt\_B\_values} \right) \]

## Details of Testing Steps

## Conclusion of Testing Results

## Final Conclusion
