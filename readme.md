
# The official implementation for the paper "Accurate and Lightweight Learning for Specific Domain Image-Text Retrieval" from the 2024 ACM Multimedia Conference. 

The paper introduces two innovations: the MLCE loss and the SPDS strategy.

**The current code is not complete; the SPDS section is still being organized...**

The version of the code we have open-sourced is an updated version from the paper. In the open-source version, the compatible versions of Torch, CUDA, and cuDNN have been updated. We recommend using Torch 1.9.1 with CUDA 11.1 (cu111).

## Here, we use the remote sensing image-text retrieval task as an example to demonstrate how to use the code.

  ### (1) Data Preparation: Users should modify their data paths in lines 30-38 of adds/params.py (using RSITMD as an example in the file).
  ### (2) Model Weights Preparation: Users should modify the path to the pre-trained weights of the CLIP model in lines 45-49. We provide the CLIP weights "ViT-B-16.pt", available at: https://drive.google.com/file/d/1mMh-KwED4jbDJjSb2dIkF_um8yCxmCEr/view?usp=drive_link 
Users can also download it from the official CLIP implementation or Hugging Face.
  ### (3) MLCE Loss Function Instructions: Users can view the source code in lines 22-32 of the loss.py file. Set the combination coefficient for the MLCE loss on line 19.

## Recommended Hyperparameter Settings:
In a Torch 1.9.1 with CUDA 11.1 (cu111) environment, we recommend setting the combination coefficient of the MLCE loss function to **30** for the RSITMD dataset and **10** for the RSICD dataset. 
We found that the optimal hyperparameters might vary with different PyTorch and CUDA versions. Therefore, we recommend users perform hyperparameter tuning in their own environments. 

## Run: python -u train.py



