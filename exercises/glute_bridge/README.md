# 🍑 Glute Bridge Exercise Correction AI

An AI-powered system that analyzes glute bridge form in real-time using computer vision and provides instant feedback for perfect technique. 🎯

## ✨ Features

- **📹 Real-time pose analysis** through webcam
- **🔍 Intelligent form correction** using pose detection
- **⚡ Instant feedback** on technique accuracy
- **🔢 Automatic rep counting** with hold time tracking
- **⏱️ Hold duration monitoring** for optimal muscle engagement
- **📊 Visual progress indicators** and feedback display

## 🚀 Quick Start

### Prerequisites
```bash
pip install opencv-python mediapipe pandas numpy scikit-learn matplotlib seaborn onnx onnxruntime xgboost tqdm
```

### Usage Steps
1. **📚 Training Pipeline** - Run notebooks 1-4 in sequence to train the model
2. **🏋️ Live Exercise** - Use notebook 5 for real-time form correction
3. **📸 Camera Setup** - Position yourself sideways to webcam showing full body profile
4. **🎮 Controls** - Press 'q' to quit the application

> **💡 Pro Tip:** Ensure good lighting and clear background for optimal pose detection

## 📓 Notebook Workflow

| Order | Notebook | Purpose | Description |
|-------|----------|---------|-------------|
| 1️⃣ | `1_exploratory_data_analysis.ipynb` | 📊 **Data Analysis** | Analyze landmark data and pose distributions |
| 2️⃣ | `2_data_preprocessing_and_model_training.ipynb` | 🤖 **Model Training** | Feature extraction, data augmentation, ML training |
| 3️⃣ | `3_model_inference.ipynb` | 🧪 **Model Testing** | Performance evaluation with detailed metrics |
| 4️⃣ | `4_deploy.ipynb` | 🚀 **Model Deployment** | Convert to ONNX format for optimization |
| 5️⃣ | `5_live_camera_detection.ipynb` | **🏋️ Live Application** | **Main app - Start here for exercise!** |

## 🎮 User Interface

### Visual Feedback
- **🟢 Green text** → Correct form ✅
- **🔴 Red/Orange text** → Form correction needed ⚠️
- **📊 Rep counter** → Automatic counting display
- **⏱️ Progress bar** → Hold duration indicator
- **💬 Live messages** → "Lift your hips", "Hold it...", "Great hold!" 🎉

### Controls
- **'q'** → Quit application ❌

## 🧠 Technical Overview

### AI Architecture
The system uses **MediaPipe** for pose detection and extracts key biomechanical features:

- **🦴 Hip angles** - Primary indicator for glute activation
- **🦵 Knee angles** - Stability and positioning analysis  
- **🏃 Body alignment** - Straight line from knees to shoulders
- **📏 Hip deviation** - Distance from optimal position
- **📐 Geometric features** - Comprehensive form analysis

### Machine Learning
A **🌲 Random Forest model** (with alternative algorithms) analyzes extracted features to classify exercise form accuracy.

## 📁 Project Structure

```
glute_bridge/
├── 📊 data/                    # Training datasets (train.csv, test.csv)
├── 🤖 models/                  # Trained AI models
│   ├── pkl/                   # Scikit-learn models (.pkl files)
│   └── onnx/                  # Optimized deployment models (.onnx files)
├── 📓 notebooks/              # Jupyter notebooks (main workflow)
└── 📖 README.md               # Project documentation
```

## 🏋️ Exercise Guidelines

### ✅ Correct Form
- Lie on back with knees bent at 90° 🦵
- Feet flat, hip-width apart 👣
- Squeeze glutes and lift hips up 🍑
- Maintain straight line: knees → hips → shoulders 📏
- Hold top position (1.5 seconds) ⏱️
- Lower with control ⬇️

### ❌ Common Mistakes
- Excessive back arching 🚫
- Insufficient hip elevation 🚫  
- Rushed movement patterns 🚫
- Poor glute engagement 🚫

## 🔧 Troubleshooting

### Camera Issues
- **📹 No detection:** Check webcam permissions and lighting
- **🖼️ Poor tracking:** Ensure clear background and full body visibility
- **👤 Position:** Stand/lie sideways to camera for profile view

### Model Performance
- **🤖 Retraining:** Run notebook 2 with adjusted parameters
- **⚙️ Algorithm:** Try different models (RF, XGB, LR)
- **🔄 Data:** Increase augmentation for better generalization

### Environment Setup
- **📦 Dependencies:** Verify all packages are correctly installed
- **🔐 Permissions:** Ensure camera access is granted
- **💡 Lighting:** Bright, even lighting improves detection accuracy
- **🖥️ Resources:** Monitor CPU/GPU usage during training

## 🚀 Advanced Configuration

### Model Training Process
1. **Data Preprocessing** → Feature extraction and normalization
2. **Data Augmentation** → Rotation, translation, scaling transformations  
3. **Model Training** → Multiple ML algorithms with cross-validation
4. **Model Evaluation** → Performance metrics and validation
5. **Model Selection** → Best performer converted to ONNX format

### Performance Optimization
- Use ONNX runtime for faster inference
- Adjust detection confidence thresholds
- Optimize video frame processing rate
- Fine-tune model hyperparameters

---

**🎉 Ready to perfect your glute bridges?** Open `5_live_camera_detection.ipynb` and start training! 🍑✨

**❓ Need help?** Check individual notebook comments for detailed explanations. 📚