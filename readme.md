# EEG Seizure Detection Repository

Github repo by Ryan Gabriel.  
Credits to Noor-Al-Shawa, Dr Jac Fredo A.R. from IIT BHU school of biomedical engineering.

## Overview

This repository contains code and notebooks for EEG seizure detection using various deep learning models. The project focuses on analyzing EEG spectrograms (MEL, CWT, STFT) to classify seizure events. It includes hyperparameter optimization using Optuna, model training notebooks for different architectures (LSTM, Transformers, Vision Transformers, Swin Transformers, Transformer-CNN hybrids, EEGNet), and visualization tools for interpretability (e.g., saliency maps, Grad-CAM).

The repository is structured to support both 2-class and 3-class classification tasks, with cross-validation across multiple folds. Models are trained on preprocessed EEG data stored in numpy arrays and pickle files.

## Data
The notebooks load data from external paths (e.g., D:\PYTHONIG\newwindow\numpy\ORIGINAL DATA\MEL), which include numpy arrays for EEG folds (eeg_fold_1.npy to eeg_fold_5.npy), labels, and patient IDs. Data is preprocessed into spectrograms.

### Spectrograms
EEG signals are transformed into spectrograms for analysis, as raw time-series data is challenging for image-based models. The repository uses three main types:

- **MEL Spectrograms**: Based on the Mel scale, which approximates human auditory perception. MEL spectrograms emphasize lower frequencies and are computed using a filterbank that warps frequencies to match human hearing. Useful for capturing harmonic structures in EEG signals related to seizures.

- **CWT (Continuous Wavelet Transform) Spectrograms**: Decomposes signals into time-frequency representations using wavelets. CWT provides high resolution in both time and frequency domains, ideal for detecting transient events like seizures. It uses mother wavelets (e.g., Morlet) to analyze signal variations.

- **STFT (Short-Time Fourier Transform) Spectrograms**: Divides the signal into short segments and applies FFT to each, creating a time-frequency matrix. STFT balances time and frequency resolution, commonly used for stationary signal analysis but adapted here for EEG dynamics.

These spectrograms are 2D representations (e.g., 100x100 pixels) fed into models like ViT or CNNs.

## Models

This section provides extensive explanations of the deep learning models used in the repository for EEG seizure detection. Each model is described in detail, including its definition, working principles, approximate number of parameters, and code snippets from the implementation.

### LSTM with Attention

**Definition**: Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) designed to handle sequential data by mitigating the vanishing gradient problem through specialized gates (input, forget, output). The "with Attention" variant incorporates an attention mechanism that allows the model to focus on relevant parts of the input sequence when making predictions, improving performance on tasks like time-series classification.

**Working Principles**: LSTMs process sequences step-by-step, maintaining a hidden state that captures long-term dependencies. The attention mechanism computes weights for each time step, creating a context vector that emphasizes important features. In EEG analysis, LSTMs with attention are applied to spectrogram sequences (e.g., concatenated CWT or STFT), treating them as time-series data. The model learns to attend to frequency bands or time windows indicative of seizures.

**Approximate Number of Parameters**: For a typical LSTM with attention on EEG data (e.g., input size 100x100 spectrograms flattened or sequenced), with 2 LSTM layers, hidden size 256, and attention heads, the model might have around 1-2 million parameters. This includes LSTM weights (input-to-hidden, hidden-to-hidden), attention weights, and classifier layers.

**Advantages**: Effective for sequential data, captures temporal dependencies in EEG signals. Attention improves interpretability by highlighting key time steps.

**Disadvantages**: Computationally intensive for long sequences, may struggle with very high-dimensional spectrograms without preprocessing.

**Code Snippet** (from LSTM notebooks):
```python
class LSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq_len, hidden)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden)
        return self.classifier(context)
```

This implementation shows the core LSTM with attention. In the notebooks, it's optimized with Optuna for parameters like hidden_size and num_layers, and trained on concatenated spectrograms for 2/3-class classification.

**Training Insights**: The model uses early stopping to prevent overfitting, with patient-wise cross-validation to ensure generalization. Logs show convergence after 50-100 epochs, achieving high F1-scores on seizure detection.

**Variations**: Notebooks include versions for CWT, MEL, and STFT spectrograms, with concatenated inputs for multi-modal fusion.

**Parameter Estimation**: For input_size=1000 (flattened spectrogram), hidden_size=256, num_layers=2, num_classes=2, parameters ≈ 1.5M (LSTM: ~1M, attention: ~65K, classifier: ~514).

**Use Cases in Repo**: Primarily in `optuna_hyperparameter_tuning/lstm/` and `training_notebooks/` for sequence-based EEG classification.

**Further Reading**: Based on papers like "Attention-Based LSTM for Time Series Prediction" adapted for medical signals.

This detailed explanation covers the model's architecture, implementation, and application in EEG seizure detection, providing a comprehensive understanding for researchers and practitioners.

### Standard Transformers

**Definition**: Transformers are attention-based neural networks introduced in "Attention is All You Need" by Vaswani et al. They rely entirely on self-attention mechanisms to process sequences, without recurrence or convolution. In this context, standard Transformers are used for processing spectrogram patches or sequences in EEG classification.

**Working Principles**: The model consists of encoder layers with multi-head self-attention and feed-forward networks. Positional encodings are added to input embeddings to capture order. For EEG, spectrograms are patched or sequenced, then processed through multiple encoder layers. Self-attention computes relationships between all positions, allowing the model to focus on relevant frequency-time combinations for seizure detection.

**Approximate Number of Parameters**: For a Transformer with 6 encoder layers, hidden size 512, 8 attention heads, and input vocab size ~10K (for patches), parameters can reach 60-100 million. In the repo's implementations (e.g., hidden_size=256, 2 layers), it's scaled down to ~5-10 million parameters for efficiency.

**Advantages**: Excellent at capturing long-range dependencies, parallelizable training, strong performance on sequence tasks.

**Disadvantages**: Requires large datasets, high computational cost, may overfit without regularization.

**Code Snippet** (from Transformer notebooks):
```python
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, num_classes):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1)]
        x = self.transformer(x)
        return self.classifier(x.mean(dim=1))  # Pooling
```

This shows a basic Transformer encoder. Notebooks optimize num_layers and num_heads via Optuna, using 5-fold CV for robust evaluation.

**Training Insights**: Trained with Adam optimizer, learning rate scheduling, and early stopping. Achieves high accuracy on 3-class tasks, with logs tracking attention weights for interpretability.

**Variations**: Includes 3-class versions and concatenated spectrogram inputs.

**Parameter Estimation**: For hidden_dim=256, num_layers=2, num_heads=4, input_dim=1000, parameters ≈ 2.5M (embeddings: 256K, transformer: 2M, classifier: 514).

**Use Cases in Repo**: In `optuna_hyperparameter_tuning/Transformer/` and training notebooks for attention-based classification.

**Further Reading**: Original Transformer paper, adapted for vision/sequence tasks in medical imaging.

This extensive overview details the Transformer's mechanics, scalability, and EEG-specific adaptations.

### Vision Transformers (ViT)

**Definition**: Vision Transformers (ViT) apply the Transformer architecture to images by dividing them into patches and treating patches as tokens in a sequence. Introduced by Dosovitskiy et al., ViT has revolutionized computer vision by achieving state-of-the-art results without convolutions.

**Working Principles**: An image (or spectrogram) is split into fixed-size patches, each linearly embedded into a vector. A class token ([CLS]) is prepended, and positional embeddings are added. The sequence is processed by Transformer encoders. For EEG, spectrograms are treated as 2D images, patched (e.g., 10x10 patches), and classified via the [CLS] token.

**Approximate Number of Parameters**: ViT-Base has ~86M parameters, but repo versions (e.g., patch_size=10, hidden_size=256, 2 layers) have ~1-5M parameters, scaled for EEG data.

**Advantages**: Scalable to large datasets, captures global context, strong on image-like data.

**Disadvantages**: Requires pre-training on large datasets, less inductive bias than CNNs.

**Code Snippet** (from ViT notebooks):
```python
def patchify(data, n_patches):
    n, c, h, w = data.shape
    patches = torch.zeros(n, n_patches**2, (c*h*w)//(n_patches**2))
    patch_size = h // n_patches
    for idx, d in enumerate(data):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = d[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                patches[idx, i*n_patches + j] = patch.flatten()
    return patches

class ViT(nn.Module):
    def __init__(self, input_size, n_patches, hidden_size):
        super().__init__()
        self.patch_size = (input_size[1]*input_size[2]*input_size[0]) // (n_patches**2)
        self.linear_mapper = nn.Linear(self.patch_size, hidden_size)
        self.class_token = nn.Parameter(torch.rand(1, hidden_size))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches**2 + 1, hidden_size))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=4, batch_first=True), num_layers=2
        )
        self.classifier = nn.Linear(hidden_size, 2)
    
    def forward(self, x):
        patches = patchify(x, self.n_patches)
        tokens = self.linear_mapper(patches)
        tokens = torch.cat([self.class_token.expand(len(tokens), -1, -1), tokens], dim=1)
        tokens += self.pos_embed
        encoded = self.encoder(tokens)
        return self.classifier(encoded[:, 0])
```

This implements ViT for spectrograms. Notebooks include saliency hooks for interpretability.

**Training Insights**: Uses Optuna for n_patches and hidden_size, with 5-fold CV. Converges quickly, high performance on MEL spectrograms.

**Variations**: 2/3-class, different spectrograms.

**Parameter Estimation**: For n_patches=10, hidden_size=256, parameters ≈ 2M (mapper: 640K, pos_embed: 25.6K, encoder: 1.3M, classifier: 514).

**Use Cases in Repo**: Core in `optuna_hyperparameter_tuning/ViT/` and visualization.

**Further Reading**: ViT paper, applications in medical imaging.

This detailed description covers ViT's patch-based approach and EEG adaptations.

### Swin Transformers

**Definition**: Swin Transformers (Shifted Window Transformers) are hierarchical vision Transformers that use shifted windows for self-attention, enabling efficient computation and scalability. Proposed by Liu et al., they build multi-scale feature maps like CNNs but with attention.

**Working Principles**: Images are divided into windows, self-attention computed within windows, then windows shift for cross-window interactions. This creates a pyramid structure. For EEG spectrograms, Swin processes patches hierarchically, capturing local and global patterns.

**Approximate Number of Parameters**: Swin-Tiny has ~28M, but repo implementations (scaled) have ~5-10M parameters.

**Advantages**: Efficient, hierarchical features, better than ViT on small datasets.

**Disadvantages**: Complex implementation, higher memory for large images.

**Code Snippet** (simplified from Swin notebooks):
```python
# Simplified Swin block
class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))
    
    def forward(self, x, mask):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        return self.norm2(x + mlp_out)

class SwinTransformer(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        # Patch embedding, multiple Swin blocks, downsampling
        self.classifier = nn.Linear(768, num_classes)  # Example
    
    def forward(self, x):
        # Patchify, process through blocks
        return self.classifier(x)
```

Notebooks optimize window sizes and depths.

**Training Insights**: Robust to overfitting, used in 3-class tasks.

**Variations**: For CWT, MEL, STFT.

**Parameter Estimation**: ~5M for small configs.

**Use Cases in Repo**: In `optuna_hyperparameter_tuning/Swin Transformer/`.

**Further Reading**: Swin Transformer paper.

This covers the hierarchical attention mechanism.

### Transformer-CNN Hybrids

**Definition**: These models combine convolutional neural networks (CNNs) for local feature extraction with Transformers for global modeling. They leverage CNNs' inductive bias and Transformers' attention for tasks like image/sequence classification.

**Working Principles**: CNN layers extract features, then flattened or sequenced into Transformers. For EEG, CNNs process spectrograms, Transformers handle sequences. Hybrids like InceptionV3 + Transformer are used.

**Approximate Number of Parameters**: InceptionV3 has ~27M, plus Transformer ~5M, total ~30-50M, but custom hybrids in repo are smaller (~10M).

**Advantages**: Combines strengths of both, effective for visual sequences.

**Disadvantages**: Complex, higher parameters.

**Code Snippet** (from Transformer-CNN notebooks):
```python
class TransformerCNN(nn.Module):
    def __init__(self, cnn_backbone, transformer_layers, num_classes):
        super().__init__()
        self.cnn = cnn_backbone  # e.g., InceptionV3
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8), num_layers=transformer_layers
        )
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, x):
        features = self.cnn(x)  # CNN features
        features = features.view(features.size(0), -1, 768)  # Reshape for transformer
        encoded = self.transformer(features)
        return self.classifier(encoded.mean(dim=1))
```

Used in custom and InceptionV3 variants.

**Training Insights**: Optuna tunes CNN-Transformer integration.

**Variations**: Custom, Inception-based.

**Parameter Estimation**: ~10M for hybrid.

**Use Cases in Repo**: In `optuna_hyperparameter_tuning/Transformer-CNN/`.

**Further Reading**: Hybrid models in vision.

This concludes the extensive model explanations.

### EEGNet

**Definition**: EEGNet is a compact convolutional neural network specifically designed for EEG-based brain-computer interfaces. It uses depthwise and separable convolutions to efficiently model spatial and temporal features in EEG signals.

**Working Principles**: EEGNet employs temporal convolutions to capture frequency-specific responses, followed by depthwise spatial convolutions to learn spatial filters. It includes separable convolutions for efficiency. For spectrograms, it's adapted to process 2D representations with minimal parameters.

**Approximate Number of Parameters**: Typically ~2-5K parameters, making it very lightweight compared to other models.

**Advantages**: Low parameter count, fast training, effective for EEG classification.

**Disadvantages**: May lack depth for complex patterns.

**Code Snippet** (from EEGNET notebooks):
```python
class EEGNet(nn.Module):
    def __init__(self, num_classes=3, num_channels=20, num_timepoints=5120):
        super().__init__()
        self.T = num_timepoints
        self.conv1 = nn.Conv2d(1, 16, (1, num_channels), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        self.fc1 = nn.Linear(2560, num_classes)
    
    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        x = x.reshape(-1, 4*2*x.size(3))
        x = F.sigmoid(self.fc1(x))
        return x
```

**Training Insights**: Optimized with Optuna, used for 3-class classification.

**Variations**: 2/3-class.

**Parameter Estimation**: ~4K parameters.

**Use Cases in Repo**: In `EEGNET/` folder.

**Further Reading**: EEGNet paper by Lawhern et al.

This adds EEGNet to the model suite.

## Repository Structure

```
eeg_seizure/
├── readme.md                           # This README file
├── EEGNET/                             # EEGNet model notebooks
│   ├── EEGNET OPTUNA.ipynb             # Optuna optimization for EEGNet
│   └── EEGNET3CLASS.ipynb              # 3-class training with EEGNet
├── optuna_hyperparameter_tuning/       # Hyperparameter optimization notebooks using Optuna
│   ├── lstm/                           # LSTM models with attention
│   │   ├── CWT LSTMAT CONCATENATED SPECTROGRAMS OPTUNA.ipynb
│   │   ├── STFT LSTMAT CONCATENATED SPECTROGRAMS OPTUNA copy.ipynb
│   │   ├── cwtlogs.txt                 # Logs for CWT LSTM optimization
│   │   └── stftlogslstmat.txt          # Logs for STFT LSTM optimization
│   ├── Swin Transformer/               # Swin Transformer models
│   │   ├── opt1fold_swinCWT.ipynb      # 1-fold optimization for CWT
│   │   ├── opt1fold_swinSTFT.ipynb     # 1-fold optimization for STFT
│   │   ├── opt5fold_swin.ipynb         # 5-fold optimization
│   │   ├── optunalogCWT.txt            # CWT logs
│   │   ├── optunalogMEL.txt            # MEL logs
│   │   └── optunalogSTFT.txt           # STFT logs
│   ├── Transformer/                    # Standard Transformer models
│   │   ├── optunaconcatenated.ipynb    # Concatenated spectrograms
│   │   ├── templog                     # Temporary log file
│   │   ├── transformer STFT logs.txt   # STFT logs
│   │   ├── Transformer3class copy.ipynb
│   │   ├── Transformer3class.ipynb     # 3-class classification
│   │   ├── transformercwtlogs.txt      # CWT logs
│   │   ├── transformermellogs.txt      # MEL logs
│   │   ├── TRANSFORMERopt5foldssCWT.ipynb  # 5-fold CWT
│   │   ├── TRANSFORMERopt5foldssMEL.ipynb  # 5-fold MEL
│   │   └── TRANSFORMERopt5foldssSTFT.ipynb  # 5-fold STFT
│   ├── Transformer-CNN/                # Transformer-CNN hybrid models
│   │   ├── cwtlog1                     # CWT log
│   │   ├── meloptunalog                # MEL log
│   │   ├── opt10folds.ipynb            # 10-fold optimization
│   │   ├── opt5foldssINDIVID.ipynb     # Individual fold optimization
│   │   ├── optunacustommodel copy.ipynb
│   │   ├── optunaincpetionv3.ipynb     # InceptionV3 based
│   │   ├── stft optuna log.txt         # STFT log
│   │   ├── TRANS-CNN_opt5foldss copy.ipynb
│   │   ├── TRANS-CNN_opt5foldss.ipynb # 5-fold TRANS-CNN
│   │   ├── TRANS-CNN_opt5foldsSTFT.ipynb # STFT variant
│   │   ├── 10FOLDS/                    # 10-fold results
│   │   │   ├── 2class.ipynb            # 2-class
│   │   │   └── 3class.ipynb            # 3-class
│   │   ├── custom/                     # Custom models
│   │   │   ├── 2class.ipynb
│   │   │   ├── 3class copy 2.ipynb
│   │   │   ├── 3class copy.ipynb
│   │   │   ├── 3class experiment.ipynb
│   │   │   ├── 3class.ipynb
│   │   │   ├── mel3class.ipynb         # MEL 3-class
│   │   │   ├── normalized2class.ipynb  # Normalized data
│   │   │   └── normalized3class.ipynb
│   │   ├── inceptionv3/                # InceptionV3 models
│   │   │   ├── 2class.ipynb
│   │   │   └── 3class.ipynb
│   └── ViT/                            # Vision Transformer models
│       ├── MELViT3classtop5.ipynb      # MEL top 5 params
│       ├── optunaconcatenated.ipynb    # Concatenated
│       ├── params.csv                  # Parameter file
│       ├── ViT3class.ipynb             # 3-class ViT
│       ├── vitcwtlogs                  # CWT logs
│       ├── vitmellogs                  # MEL logs
│       ├── ViTopt5foldssCWT.ipynb      # 5-fold CWT
│       ├── ViTopt5foldssMEL copy.ipynb # MEL copy
│       ├── ViTopt5foldssSTFT.ipynb     # 5-fold STFT
│       └── vitstftlogs.txt             # STFT logs
├── training_notebooks/                 # Training notebooks with top hyperparameters
│   ├── 2 class/                        # 2-class classification
│   │   ├── LSTMAT CWT top5 CONCATENATED 2 CLASS.ipynb
│   │   ├── LSTMAT MEL top5 CONCATENATED 2 CLASS.ipynb
│   ├── LSTMAT STFT top5 CONCATENATED 2 CLASS.ipynb
│   │   ├── TRANS-CNN 2 cls mel top5.ipynb
│   │   ├── TRANS-CNN 2 cls STFT top5.ipynb
│   │   ├── TRANS-CNN 2cls cwt top5.ipynb
│   │   ├── Transformer CWT 2class top5.ipynb
│   │   ├── Transformer MEL 2class top5 copy.ipynb
│   │   ├── Transformer STFT 2class top5 copy 2.ipynb
│   │   ├── ViT CWT 2class.ipynb
│   │   ├── ViT MEL 2class copy.ipynb
│   │   └── ViT STFT 2class.ipynb
│   └── 3 class/                        # 3-class classification
│       ├── LSTMAT cwt 3CLASS.ipynb
│       ├── LSTMAT mel 3CLASS.ipynb
│       ├── LSTMAT STFT 3CLASS.ipynb
│       ├── SWIN CWT 3class copy 2.ipynb
│       ├── SWIN CWT 3class set1.ipynb
│       ├── SWIN MEL 3class.ipynb
│       ├── SWIN STFT 3class copy.ipynb
│       ├── TRANS-CNN cwt top5params.ipynb
│       ├── TRANS-CNN mel top5params.ipynb
│       ├── TRANS-CNN STFT top5params.ipynb
│       ├── Transformer STFTtop5.ipynb
│       ├── TransformerCWTtop5.ipynb
│       ├── TransformerMELtop5.ipynb
│       ├── ViT CWT 3classtop5.ipynb
│       ├── ViT MEL 3classtop5.ipynb
│       └── ViT STFT 3classtop5.ipynb
└── visualization_notebooks/            # Notebooks for model interpretability
    ├── trysaliency.ipynb               # Saliency map generation for ViT
    └── GRADCAM/                        # Grad-CAM visualizations
        └── ViTGradcam.ipynb            # Grad-CAM for ViT
```

## Detailed Description of Folders and Files

### EEGNET/
Contains notebooks for the EEGNet model, a lightweight CNN optimized for EEG signals.

- **EEGNET OPTUNA.ipynb**: Hyperparameter optimization using Optuna for EEGNet parameters.
- **EEGNET3CLASS.ipynb**: Training EEGNet for 3-class seizure classification with cross-validation.

### optuna_hyperparameter_tuning/
This folder contains Jupyter notebooks for hyperparameter optimization using the Optuna library. Each subfolder corresponds to a different model architecture, and notebooks perform optimization on different spectrogram types (CWT, MEL, STFT) and fold configurations.

- **lstm/**: Focuses on LSTM with attention mechanisms. Notebooks optimize for concatenated spectrograms.
- **Swin Transformer/**: Optimizes Swin Transformer models for various spectrograms and folds.
- **Transformer/**: Standard Transformer models, including 3-class variants and 5-fold cross-validation.
- **Transformer-CNN/**: Hybrid models combining Transformers and CNNs, with subfolders for different configurations (10FOLDS, custom, inceptionv3).
- **ViT/**: Vision Transformer optimizations, including parameter CSV and logs.

Log files (.txt) store optimization results and can be used to track performance metrics.

### training_notebooks/
These notebooks use the top hyperparameters found via Optuna to train final models. Divided into 2-class and 3-class subfolders.

- **2 class/**: Notebooks for binary classification (e.g., seizure vs. non-seizure).
- **3 class/**: Notebooks for multi-class classification (e.g., different seizure types).

Each notebook specifies the model type, spectrogram (CWT, MEL, STFT), and often includes "top5" indicating use of optimized parameters.

### visualization_notebooks/
Contains notebooks for interpreting model predictions.

- **trysaliency.ipynb**: Generates saliency maps for Vision Transformer models to highlight important regions in EEG spectrograms.
- **GRADCAM/ViTGradcam.ipynb**: Implements Grad-CAM for ViT to visualize class activation maps.

## Usage
1. Run Optuna notebooks to find optimal hyperparameters.
2. Use training notebooks with top parameters for final model training.
3. Visualize results with saliency and Grad-CAM notebooks.

## Dependencies
- PyTorch
- Optuna
- NumPy, Matplotlib, Scikit-learn
- Torchvision, Torcheval
- Jupyter Notebook

Install via pip: `pip install torch optuna numpy matplotlib scikit-learn torchvision torcheval`

## Credits
- Ryan Gabriel (Repository maintainer)
- Noor-Al-Shawa
- Dr. Jac Fredo A.R. (IIT BHU School of Biomedical Engineering)

## License
[Add license if applicable]

## Code Explanations

This section provides detailed explanations of key concepts and code implementations used in the repository, including Optuna for hyperparameter optimization, sample training code, Grad-CAM for visualization, and saliency maps.

### Optuna Code
**Concept**: Optuna is an open-source hyperparameter optimization framework that automates the search for the best hyperparameters in machine learning models. It uses techniques like Tree-structured Parzen Estimator (TPE) to efficiently explore the hyperparameter space, minimizing the number of trials needed to find optimal settings. In this repository, Optuna is used to tune parameters such as learning rate, batch size, number of layers, and attention heads for various deep learning models.

**Implementation**: The Optuna notebooks (e.g., in `optuna_hyperparameter_tuning/`) define an objective function that trains a model with suggested hyperparameters, evaluates it on validation data, and returns a metric (e.g., accuracy or loss). Optuna then iteratively suggests new parameters to maximize or minimize the objective.

Example pseudocode from an Optuna notebook:
```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Define and train model
    model = ViT(input_size, n_patches, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Training loop...
    
    # Return validation accuracy
    return val_accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(study.best_params)
```

### Sample Training Code
**Concept**: Training code refers to the process of teaching a machine learning model on a dataset by adjusting its parameters to minimize prediction errors. In deep learning, this involves forward passes to compute predictions, backward passes to calculate gradients, and optimizer updates. The training notebooks in `training_notebooks/` implement this for EEG classification, using cross-validation, early stopping, and evaluation metrics.

**Implementation**: Notebooks load preprocessed data, define models with optimized hyperparameters, and run training loops with loss computation, backpropagation, and validation. They often include data augmentation, patient-wise splitting to avoid data leakage, and metrics like accuracy, F1-score, and balanced accuracy.

Example pseudocode from a training notebook:
```python
# Load data
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define model, loss, optimizer
model = ViT((20, 100, 100), 10, 256)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
    
    # Early stopping check
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        break
```

### Grad-CAM Code
**Concept**: Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique for visualizing which regions of an input image (or spectrogram) are most important for a model's prediction. It computes gradients of the predicted class score with respect to feature maps from the last convolutional layer, then weights and combines them to produce a heatmap. This helps interpret model decisions, especially in medical imaging like EEG spectrograms.

**Implementation**: In `visualization_notebooks/GRADCAM/ViTGradcam.ipynb`, Grad-CAM is adapted for Vision Transformers by using attention weights or gradients from the transformer layers. The code hooks into the model to capture gradients and activations, then generates heatmaps overlaid on spectrograms.

Example pseudocode:
```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, class_idx):
        self.model.zero_grad()
        pred = self.model(input_image)
        pred[:, class_idx].backward()
        
        # Compute Grad-CAM
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam
```

### Saliency Code
**Concept**: Saliency maps highlight the pixels or regions in an input that have the highest impact on the model's output. They are computed by taking the absolute value of the gradients of the output with respect to the input, showing where small changes would most affect the prediction. This is useful for understanding model focus in tasks like EEG analysis.

**Implementation**: In `visualization_notebooks/trysaliency.ipynb`, saliency is implemented for ViT models by enabling gradients on the input, performing a forward pass, computing the gradient of the predicted class score, and normalizing the absolute gradients. The notebook also averages saliency maps across categories for comparative visualization.

Example pseudocode:
```python
def saliency_map(input_tensor, model, class_idx):
    model.eval()
    input_tensor.requires_grad_(True)
    
    output = model(input_tensor)
    score = output[0, class_idx]
    
    model.zero_grad()
    score.backward()
    
    saliency = torch.abs(input_tensor.grad[0]).sum(dim=0)  # Sum over channels
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    return saliency.detach().cpu().numpy()
```

## Repository Structure