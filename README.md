# ViT vs. ResNet for Image Classification

This project compares the performance of a **Vision Transformer (ViT-Base-16)** and a **ResNet-50** model on the CIFAR-100 dataset. It was developed using **PyTorch**, `timm`, and `torchvision` and is designed to be run inside **PyCharm** or similar IDEs.

---

## 📁 Project Structure

```
project-directory/
- vit_resnet_comparison.py   # Main Python script (training + evaluation)
- accuracy_comparison.png    # Saved accuracy plot (output)
- requirements.txt           # Project dependencies
- README.md                  # Project documentation
- data/                      # Automatically created to store CIFAR-100 dataset
```

---

## 🎯 Objective

* Fine-tune ViT-Base-16 and ResNet-50 on CIFAR-100
* Compare accuracy, training time, and model size
* Analyze differences in performance and architecture

---

## 🗂 Dataset: CIFAR-100

* 60,000 images (100 classes)
* Size: 32x32 pixels (resized to 224x224)
* 50,000 for training, 10,000 for testing

---

## ⚙️ Model Details

### ✅ Vision Transformer (ViT-Base-16)

* Loaded from `timm`
* Head modified for 100 classes
* Optimizer: Adam
* Learning rate: 5e-5

### ✅ ResNet-50

* Loaded from `torchvision.models`
* Final FC layer changed to 100 classes
* Optimizer: SGD with momentum
* Learning rate: 1e-3

---

## 🧪 Training Setup

* Batch Size: 32
* Epochs: 30
* Data augmentations: RandomCrop, HorizontalFlip
* Evaluation: Top-1 Accuracy
* Logging with progress prints every 50 batches

---

## 📊 Evaluation Metrics

* Top-1 accuracy on test set
* Training accuracy plotted per epoch
* Summary printed at end

### Example Output:

```
Summary:
ViT Test Accuracy: 85.32%
ResNet Test Accuracy: 80.42%
ViT has more parameters and is slower but performs better on CIFAR-100.
ResNet is faster and lighter, but with slightly lower accuracy.
```

---

## 📉 Visualizations

* Training accuracy vs. epochs plot saved as `accuracy_comparison.png`

---

## 📦 Installation & Run

1. Clone the repo:

```bash
git clone https://github.com/hiwea/vit-vs-resnet.git
cd vit-vs-resnet
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the project:

```bash
python vit.py
```

---

## 🧾 requirements.txt

The file contains:

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.2
matplotlib>=3.7.0
```

Use this to install all required libraries at once.

---

## 📚 Notes

* You can add Grad-CAM or attention maps for visualization
* For robustness, try testing with noisy images or adversarial examples

---

## 🙋‍♂️ Author

* **Hiwa Aziz Abbas**
* Project submitted as part of a Computer Vision homework assignment

---

## 📜 License

This project is licensed for academic use.
