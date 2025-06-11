# ViT vs. ResNet for Image Classification - PyCharm Version

"""


1. **Dataset Selection**
   - **Chosen Dataset:** CIFAR-100
   - **Reason:** Balanced, 100 fine-grained classes, small images, fast training

2. **Data Preparation**
   - Resize to 224x224
   - Normalize images (mean=0.5, std=0.5)
   - Augmentations: Random crop, horizontal flip
   - Used built-in train/test split

3. **Model Loading and Modification**
   - ViT: Pretrained 'vit_base_patch16_224' from `timm`, head replaced with `nn.Linear(..., 100)`
   - ResNet-50: Pretrained from `torchvision.models`, `fc` layer replaced with `nn.Linear(..., 100)`

4. **Fine-tuning Settings**
   - **ViT**: Adam optimizer, lr=5e-5, weight_decay=1e-4, full fine-tuning
   - **ResNet**: SGD optimizer, lr=1e-3, momentum=0.9, weight_decay=1e-4
   - Batch size: 32, Epochs: 30
   - Early stopping considered (not implemented here for simplicity)

5. **Evaluation Metrics**
   - Top-1 test accuracy: printed in final output
   - Training speed: Observed by epoch prints
   - Model size: ViT ~86M, ResNet ~25M parameters (approx.)
   - FLOPs: Not measured (needs special libraries)
   - Accuracy/loss curves: Plotted and saved as PNG

6. **Robustness Testing** (Optional)
   - Not implemented in this code, but could add Gaussian noise to test images to measure accuracy drop

7. **Interpretability**
   - Optional (not implemented):
     - ViT: visualize attention maps
     - ResNet: use Grad-CAM to generate saliency maps

8. **Deliverables**
   - This script (PyCharm compatible)
   - Graph accuracy_comparison.png
   - Summary printed after training


"""
