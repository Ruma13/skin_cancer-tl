# Early Detection of Skin cancer using DenseNet121 model+Fusion+CBAM+ Transfer learning+ custom cnn layers
#Project Overview

This project focuses on the early detection of skin cancer using deep learning techniques. It leverages DenseNet121 with CBAM (Convolutional Block Attention Module), model fusion, pre-pooling features, and early stopping strategies to improve diagnostic accuracy. Multiple architectures were trained and evaluated to ensure robustness and high reliability in real-world medical diagnosis.

The goal is to provide an automated pipeline for detecting and classifying skin lesions, supporting dermatologists in early detection and reducing misdiagnosis.

#Features

Training of DenseNet121 + CBAM for feature extraction and classification.

Implementation of fusion techniques combining multiple deep learning models for improved performance.

Pre-pooling features extraction to enhance model performance.

Early stopping to prevent overfitting and optimize training.

Evaluation of multiple baseline models: ResNet50, VGG16, VGG19, InceptionV3, MobileNetV2, DenseNet121, and more.

Calculation of train/validation/test metrics, including accuracy, loss, and AUC.

Visualization of training curves, confusion matrices, and Grad-CAM++ heatmaps for model explainability.

#Dataset

This project uses the HAM10000 dataset, which contains images of various skin lesion types, including melanoma, nevus, and benign keratosis.

Dataset source: HAM10000 Dataset

Directory structure used in this project:

dataset/
├── train/
│   ├── melanoma/
│   ├── nevus/
│   └── ...
└── test/
    ├── melanoma/
    ├── nevus/
    └── ...

#Installation

Clone this repository:

git clone https://github.com/<your-username>/skin-cancer-early-detection.git
cd skin-cancer-early-detection


Install the required Python packages (tested on Python 3.10+):

pip install -r requirements.txt


#Recommended packages include:

torch, torchvision

tensorflow / keras

numpy, pandas

matplotlib, seaborn

scikit-learn

#Usage
1. Prepare the Dataset

Organize images in train and test folders.

Ensure class-wise subdirectories exist.

2. Train Models

To train DenseNet121 + CBAM with fusion and pre-pooling:

python train_densenet_cbam_fusion.py --dataset_path "dataset/"


Training includes data augmentation, class balancing, and early stopping.

3. Evaluate Models

Evaluate individual models or fused models:

python evaluate_models.py --model_path "saved_models/densenet_cbam_fusion.pth"


#Outputs include:

Accuracy and loss plots

Confusion matrix

AUC score

4. Grad-CAM++ Visualization
python grad_cam_plus.py --model_path "saved_models/densenet_cbam_fusion.pth" --image "sample_image.jpg"


Generates heatmaps showing regions influencing model decisions.

#Experiments & Results
Model	Accuracy	Validation Loss	AUC
ResNet50	0.92	0.21	0.95
VGG16	0.90	0.25	0.93
DenseNet121 + CBAM + Fusion	0.96	0.15	0.98
MobileNetV2	0.89	0.27	0.92

Training plots, loss curves, and Grad-CAM++ visualizations are stored in /plots.

Fusion and attention mechanisms significantly improve early detection performance.

Model Saving

Trained PyTorch models saved as .pt files.

Can also convert to .h5 for Keras compatibility:

torch.save(model.state_dict(), "model.pt")


Models compatible with Grad-CAM++ visualizations.

References

Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. CVPR.

Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. ECCV.

Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions.

Notes

GPU is recommended for training (NVIDIA GPU with CUDA support).

Adjust hyperparameters (batch size, learning rate, epochs) in config.py.

Ensure dataset paths match the paths used in the code.
