# Computer Vision Using Deep Learning - Practicals 6 to 10
## Detailed Descriptions and Explanations

**Student:** Narola Sarth Dharmeshbhai  
**Roll No:** 23BCE194

---

## PRACTICAL 6 - Neural Networks and Feature Extraction

### Practical 6.1: Cat vs Non-Cat Classification with Deep Neural Networks

**Objective:** Build and train a deep neural network from scratch for binary image classification.

**Key Concepts:**
- Deep neural network architecture design
- Binary classification using sigmoid activation
- Image preprocessing and normalization
- Training with Adam optimizer

**Implementation Details:**
1. **Dataset:** Uses h5py format files (train_catvsnoncat.h5, test_catvsnoncat.h5)
2. **Network Architecture:**
   - Input Layer: Flattened image features (64×64×3 = 12,288 features)
   - Hidden Layer 1: 20 neurons with ReLU activation
   - Hidden Layer 2: 7 neurons with ReLU activation
   - Hidden Layer 3: 5 neurons with ReLU activation
   - Output Layer: 1 neuron with Sigmoid activation (probability output)

3. **Training Configuration:**
   - Optimizer: Adam with learning rate 0.0075
   - Loss Function: Binary Cross-Entropy
   - Iterations: 3000
   - Image normalization: Pixel values divided by 255

4. **Key Functions:**
   - `load_dataset()`: Loads training and test data from h5 files
   - `classify()`: Converts probabilities to binary predictions (threshold 0.5)
   - `show_dataset()`: Visualizes predictions on a grid of images

**Learning Outcomes:**
- Understanding multi-layer neural network architecture
- Implementing forward and backward propagation in PyTorch
- Visualizing model predictions on image data

---

### Practical 6.2: Image Feature Extraction and Classification

**Objective:** Extract handcrafted features from images and train a classifier for agricultural image recognition.

**Key Concepts:**
- Image chunking and spatial feature extraction
- RGB histogram analysis
- Feature engineering for image classification
- Handling imbalanced datasets

**Implementation Details:**
1. **Feature Extraction Process:**
   - Divide each image into 16 chunks (4×4 grid)
   - For each chunk, compute histogram for R, G, B channels
   - Extract maximum intensity value (argmax) from each histogram
   - Total features per image: 16 chunks × 3 channels = 48 features

2. **Dataset Creation:**
   - `createCSV()` function processes all images in a folder
   - Extracts features and labels (agricultural=1, non-agricultural=0)
   - Saves to CSV file for efficient loading

3. **Neural Network Architecture:**
   - Input Layer: 48 features
   - Hidden Layer: 8 neurons with ReLU
   - Dropout: 0.7 (high dropout for regularization)
   - Output Layer: 2 classes (binary classification)

4. **Training Configuration:**
   - Train-test split: 70-30 with stratification
   - Feature scaling: StandardScaler normalization
   - Optimizer: Adam (lr=0.002, weight_decay=1e-3)
   - Batch size: 8
   - Epochs: 7

**Learning Outcomes:**
- Feature engineering techniques for images
- Creating custom datasets from raw images
- Handling overfitting with dropout and regularization

---

### Practical 6.3: SIFT Features with Deep Learning

**Objective:** Combine traditional computer vision (SIFT) with deep learning for image classification.

**Key Concepts:**
- SIFT (Scale-Invariant Feature Transform) keypoint detection
- Hybrid approach: traditional CV + deep learning
- Deeper network architecture with regularization
- Extended training for better convergence

**Implementation Details:**
1. **SIFT Feature Visualization:**
   - Uses OpenCV's SIFT detector
   - Detects and visualizes keypoints on 36 sample images
   - Displays rich keypoints showing scale and orientation

2. **Deep Neural Network Architecture:**
   - Input Layer: 12,288 features (flattened 64×64×3 images)
   - Hidden Layer 1: 256 neurons with ReLU + Dropout(0.5)
   - Hidden Layer 2: 128 neurons with ReLU + Dropout(0.5)
   - Hidden Layer 3: 64 neurons with ReLU
   - Output Layer: 1 neuron with Sigmoid

3. **Training Configuration:**
   - Optimizer: Adam with learning rate 0.0005 (lower for stability)
   - Loss: Binary Cross-Entropy
   - Epochs: 1000 (extended training)
   - Tracks both loss and accuracy every 100 epochs

4. **Evaluation:**
   - Computes final training accuracy
   - Tests on separate test set
   - Reports both training and test performance

**Learning Outcomes:**
- Understanding SIFT feature detection
- Building deeper networks with proper regularization
- Monitoring training progress over extended epochs
- Evaluating model generalization

---

## PRACTICAL 7 - CNN Training with Different Optimizers

**Objective:** Implement CNN architecture and compare performance of different optimization algorithms across multiple datasets.

**Key Concepts:**
- Convolutional Neural Networks (CNN)
- Optimization algorithms comparison
- Multi-dataset training
- Model architecture design

**Implementation Details:**

### 1. CNN Architecture
```
Input: (batch, channels, height, width)
├── Conv2d(1→32, kernel=3, padding=1) + ReLU + MaxPool(2×2)
├── Conv2d(32→64, kernel=3, padding=1) + ReLU + MaxPool(2×2)
├── Flatten
├── Linear(64×7×7 → 128) + ReLU
└── Linear(128 → 10)
```

**For CIFAR-10 (3 channels, 32×32):**
```
├── Conv2d(3→32, kernel=3, padding=1) + ReLU + MaxPool(2×2)
├── Conv2d(32→64, kernel=3, padding=1) + ReLU + MaxPool(2×2)
├── Flatten
├── Linear(64×8×8 → 128) + ReLU
└── Linear(128 → 10)
```

### 2. Datasets Used
- **MNIST:** Handwritten digits (28×28, grayscale)
  - Normalization: mean=0.1307, std=0.3081
- **Fashion-MNIST:** Clothing items (28×28, grayscale)
  - Normalization: mean=0.2860, std=0.3530
- **CIFAR-10:** Natural images (32×32, RGB)
  - Normalization: mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)

### 3. Optimizers Compared

**Adam Optimizer:**
- Learning rate: 0.001
- Adaptive learning rates per parameter
- Best for: Fast convergence, good default choice

**SGD with Momentum:**
- Learning rate: 0.01
- Momentum: 0.9
- Best for: Better generalization, escaping local minima

**RMSprop:**
- Learning rate: 0.001
- Adaptive learning rates
- Best for: Non-stationary objectives

**Adagrad:**
- Learning rate: 0.01
- Accumulates squared gradients
- Best for: Sparse data, different learning rates per feature

### 4. Training Process
- Epochs: 2 per optimizer (for quick comparison)
- Batch size: 64
- Loss function: CrossEntropyLoss
- Device: GPU if available, else CPU

### 5. Model Persistence
- Saves trained CIFAR-10 model: `cifar_cnn.pth`
- Can be loaded for transfer learning in later practicals

**Learning Outcomes:**
- Understanding CNN architecture and convolution operations
- Comparing optimizer behaviors and convergence patterns
- Working with multiple standard datasets
- Model saving and loading techniques

---

## PRACTICAL 8 - Transfer Learning and Fine-Tuning

**Objective:** Apply transfer learning techniques to adapt a pre-trained model to new datasets using two different strategies.

**Key Concepts:**
- Transfer learning fundamentals
- Feature extraction vs fine-tuning
- Layer freezing techniques
- Domain adaptation

**Implementation Details:**

### 1. Base Model
- Pre-trained on CIFAR-10 (10 classes)
- Architecture: Same CNN from Practical 7
- Loads weights from `cifar_cnn.pth`

### 2. Target Datasets

**SVHN (Street View House Numbers):**
- 32×32 RGB images of house numbers
- 10 classes (digits 0-9)
- Similar to CIFAR-10 in size but different domain

**CIFAR-100:**
- 32×32 RGB images
- 100 classes (fine-grained categories)
- Same domain as CIFAR-10 but more classes

### 3. Transfer Learning Strategies

#### Strategy 1: Frozen Layers (Feature Extraction)
```python
# Copy all weights except final layer
pretrained_dict = {k: v for k, v in pretrained_dict.items() if "fc2" not in k}

# Freeze all layers except fc2
for name, param in model.named_parameters():
    if "fc2" not in name:
        param.requires_grad = False
```

**Characteristics:**
- Only trains the final classification layer
- Feature extractor remains fixed
- Faster training, less memory
- Good when: Limited data, similar domains

#### Strategy 2: Fine-Tuning (Full Network Training)
```python
# Copy weights (except final layer)
# Keep all parameters trainable
for param in model.parameters():
    param.requires_grad = True
```

**Characteristics:**
- Trains all layers end-to-end
- Adapts features to new domain
- Slower training, more memory
- Good when: More data available, different domains

### 4. Training Configuration
- Epochs: 5 for each strategy
- Optimizer: Adam (lr=0.001)
- Batch size: 64
- Loss: CrossEntropyLoss

### 5. Experiments Conducted
1. SVHN with frozen layers
2. SVHN with fine-tuning
3. CIFAR-100 with frozen layers
4. CIFAR-100 with fine-tuning

### 6. Key Observations
- **Frozen approach:** Faster, works well when domains are similar
- **Fine-tuning:** Better accuracy, adapts to new domain characteristics
- **CIFAR-100:** More challenging due to 100 classes vs 10

**Learning Outcomes:**
- Understanding when to use transfer learning
- Implementing layer freezing in PyTorch
- Comparing feature extraction vs fine-tuning
- Adapting models to different domains and class counts

---

## PRACTICAL 9 - U-Net for Medical Image Segmentation

**Objective:** Implement U-Net architecture for pixel-wise segmentation of blood vessels in retinal images.

**Key Concepts:**
- Semantic segmentation
- U-Net architecture with skip connections
- Medical image processing
- Dice coefficient metric
- Encoder-decoder networks

**Implementation Details:**

### 1. Dataset: Retinal Vessel Segmentation
- **Images:** Retinal fundus photographs (resized to 512×512)
- **Masks:** Binary masks (0=background, 1=vessel)
- **Training samples:** Images 21-36
- **Validation samples:** Images 37-40
- **Test samples:** Images 01-20

### 2. U-Net Architecture

#### Encoder (VGG16-BN backbone)
```python
- Pre-trained VGG16 with Batch Normalization
- Extracts features at multiple scales
- Initially frozen, then fine-tuned
- Outputs feature maps at 5 different resolutions
```

#### Center (Bottleneck)
```python
Conv2d(512→1024) + BatchNorm + ReLU
Conv2d(1024→1024) + BatchNorm + ReLU
```

#### Decoder (Upsampling path)
```python
For each level (5→4→3→2→1):
├── Upsample (2× nearest neighbor)
├── Conv2d (reduce channels)
├── Concatenate with encoder features (skip connection)
├── Conv2d + BatchNorm + ReLU (×3)
└── Final: Conv2d(64→2, kernel=1) for 2 classes
```

**Skip Connections:** Connect encoder layers [5, 12, 22, 32, 42] to decoder

### 3. Custom Dataset Classes

**TrainDataset:**
- Loads images 21-36
- Applies normalization (ImageNet stats)
- Converts masks to long tensor
- Resizes to 512×512

**ValDataset:**
- Loads images 37-40
- Same preprocessing as training

**TestDataset:**
- Loads images 01-20
- No ground truth masks

### 4. Training Strategy

**Phase 1: Frozen Encoder (2 epochs)**
```python
for param in pretrained_network.features.parameters():
    param.requires_grad = False
```
- Only trains center and decoder
- Faster initial training

**Phase 2: Fine-Tuning (2 epochs)**
```python
for param in pretrained_network.features.parameters():
    param.requires_grad = True
```
- Trains entire network
- Adapts encoder to medical images

### 5. Evaluation Metrics

**Dice Coefficient:**
```
Dice = 2 × |X ∩ Y| / (|X| + |Y|)
```
- Measures overlap between prediction and ground truth
- Range: 0 (no overlap) to 1 (perfect match)
- Better than accuracy for imbalanced segmentation

### 6. Training Functions

**train_one_epoch():**
- Computes loss and Dice coefficient
- Tracks running metrics per batch
- Performs backpropagation

**val_one_epoch():**
- Evaluates without gradient computation
- Computes validation Dice coefficient

### 7. Inference and Submission

**eval_one_epoch():**
- Generates predictions on test set
- Converts to run-length encoding
- Creates submission CSV file

**Run-Length Encoding:**
- Compresses binary masks
- Format: "start_pixel length start_pixel length..."
- Efficient storage for sparse masks

### 8. Visualization
- `plotres()` function displays:
  - Original image (denormalized)
  - Ground truth mask
  - Predicted segmentation mask

**Learning Outcomes:**
- Understanding U-Net architecture and skip connections
- Working with medical imaging datasets
- Implementing semantic segmentation
- Using Dice coefficient for evaluation
- Two-stage training strategy (frozen → fine-tuned)
- Run-length encoding for mask compression

---

## PRACTICAL 10 - YOLO Object Detection

**Objective:** Implement a simplified YOLO (You Only Look Once) model for real-time object detection with bounding boxes.

**Key Concepts:**
- Object detection (classification + localization)
- Grid-based detection
- Bounding box regression
- IoU (Intersection over Union)
- Multi-task loss function

**Implementation Details:**

### 1. YOLO Architecture

**Parameters:**
- S = 7 (grid size: 7×7)
- B = 4 (bounding boxes per cell)
- C = 1 (number of classes: pedestrian)

**Network Structure:**
```python
Feature Extractor:
├── Conv2d(3→32) + LeakyReLU + MaxPool
├── Conv2d(32→64) + LeakyReLU + MaxPool
├── Conv2d(64→128) + LeakyReLU + MaxPool
├── Conv2d(128→256) + LeakyReLU + MaxPool
├── Conv2d(256→512) + LeakyReLU
└── AdaptiveAvgPool2d(7×7)

Fully Connected Layers:
├── Flatten
├── Linear(512×7×7 → 1024) + LeakyReLU
└── Linear(1024 → 7×7×(C+B×5))
```

**Output Shape:** (batch, 7, 7, 21)
- 21 = 1 class + 4 boxes × 5 values
- 5 values per box: (x, y, w, h, confidence)

### 2. Bounding Box Representation

Each bounding box contains:
- **x, y:** Center coordinates (relative to cell, 0-1)
- **w, h:** Width and height (relative to image, 0-1)
- **confidence:** Objectness score (0-1)

### 3. YOLO Loss Function

**Components:**
```python
Total Loss = λ_coord × (Loss_xy + Loss_wh) 
           + Loss_conf_obj 
           + λ_noobj × Loss_conf_noobj 
           + Loss_class
```

**Loss Terms:**
1. **Coordinate Loss (Loss_xy):**
   - MSE between predicted and true (x, y)
   - Only for cells containing objects
   - Weight: λ_coord = 5

2. **Size Loss (Loss_wh):**
   - MSE between sqrt of predicted and true (w, h)
   - Square root for better small box handling
   - Weight: λ_coord = 5

3. **Confidence Loss (Object):**
   - MSE for confidence when object present
   - Weight: 1.0

4. **Confidence Loss (No Object):**
   - MSE for confidence when no object
   - Weight: λ_noobj = 0.5 (lower to handle imbalance)

5. **Class Loss:**
   - MSE for class predictions
   - Weight: 1.0

### 4. Dataset: PennFudanPed

**Custom Dataset Class:**
```python
PennFudanDataset:
├── Loads pedestrian images and segmentation masks
├── Extracts bounding boxes from masks
├── Converts to YOLO format (grid-based)
└── Limits to 20 images for quick training
```

**Bounding Box Extraction:**
1. Find unique object IDs in mask
2. Get min/max coordinates for each object
3. Convert to (x_center, y_center, width, height)
4. Normalize by image dimensions
5. Assign to appropriate grid cell

### 5. Training Configuration

- **Dataset:** 20 images from PennFudanPed
- **Batch size:** 4
- **Epochs:** 6
- **Optimizer:** Adam (lr=1e-4)
- **Image size:** 224×224
- **Device:** GPU if available

### 6. IoU Calculation

**bbox_iou() function:**
```python
# Converts center format to corner format
# Computes intersection area
# Computes union area
# Returns IoU = intersection / union
```

Used for:
- Matching predictions to ground truth
- Non-maximum suppression (NMS)
- Evaluation metrics

### 7. Inference and Visualization

**visualize_predictions():**
1. Runs model in eval mode
2. For each grid cell and bounding box:
   - Check if confidence > threshold (0.5)
   - Convert relative coordinates to pixel coordinates
   - Draw bounding box on image
3. Display results in subplot grid

**Coordinate Conversion:**
```python
# YOLO format → Pixel coordinates
bx = box[0] × image_width
by = box[1] × image_height
bw = box[2] × image_width
bh = box[3] × image_height

# Center format → Corner format
xmin = bx - bw/2
ymin = by - bh/2
xmax = bx + bw/2
ymax = by + bh/2
```

### 8. Key Differences from Full YOLO

**Simplifications:**
- Smaller network (no Darknet backbone)
- Single class detection
- No anchor boxes
- Simplified loss function
- No non-maximum suppression in training

**Educational Focus:**
- Understanding grid-based detection
- Multi-task loss implementation
- Bounding box regression
- Real-time detection concepts

**Learning Outcomes:**
- Understanding YOLO detection paradigm
- Implementing multi-task loss functions
- Working with bounding box representations
- Grid-based object detection
- Converting between coordinate systems
- Visualizing detection results
- Handling object detection datasets

---

## Summary of Key Concepts Across Practicals 6-10

### Progression of Complexity:
1. **Practical 6:** Basic neural networks and feature extraction
2. **Practical 7:** Convolutional networks and optimizer comparison
3. **Practical 8:** Transfer learning strategies
4. **Practical 9:** Advanced segmentation with U-Net
5. **Practical 10:** Object detection with YOLO

### Core Deep Learning Concepts Covered:
- Neural network architectures (FC, CNN, U-Net, YOLO)
- Optimization algorithms (Adam, SGD, RMSprop, Adagrad)
- Transfer learning and fine-tuning
- Semantic segmentation
- Object detection
- Loss functions (BCE, CrossEntropy, Dice, YOLO loss)
- Regularization techniques (Dropout, BatchNorm)
- Data preprocessing and augmentation
- Model evaluation metrics

### Practical Skills Developed:
- PyTorch implementation
- Dataset creation and loading
- Model training and evaluation
- Visualization techniques
- Working with different data formats (h5py, images, masks)
- GPU acceleration
- Model persistence (saving/loading)

---

**End of Document**
