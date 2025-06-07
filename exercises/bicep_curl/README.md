# 💪 Bicep Curl Exercise Correction AI

An AI-powered system that analyzes bicep curl form in real-time using computer vision and provides instant feedback for perfect technique. 🎯

## ✨ Features

- **📹 Real-time pose analysis** through webcam
- **🔍 Intelligent form correction** using pose detection
- **⚡ Instant feedback** on technique accuracy
- **🔢 Automatic rep counting** for both arms separately
- **🤝 Bilateral arm tracking** with independent analysis
- **📊 Visual progress indicators** and feedback display

## 🚀 Quick Start

### Prerequisites
```bash
pip install opencv-python mediapipe pandas numpy scikit-learn matplotlib seaborn onnx onnxruntime
```

### Usage Steps
1. **📚 Training Pipeline** - Run notebooks 1-4 in sequence to train the model
2. **🏋️ Live Exercise** - Use notebook 5 for real-time form correction
3. **📸 Camera Setup** - Position yourself facing webcam showing full upper body
4. **🎮 Controls** - Press 'q' to quit, 'r' to reset rep counter

> **💡 Pro Tip:** Ensure good lighting and clear background for optimal pose detection

## 📓 Notebook Workflow

| Order | Notebook | Purpose | Description |
|-------|----------|---------|-------------|
| 1️⃣ | `1_exploratory_data_analysis.ipynb` | 📊 **Data Analysis** | Analyze landmark data and pose distributions |
| 2️⃣ | `2_data_preprocessing_and_model_training.ipynb` | 🤖 **Model Training** | Feature extraction, bilateral training, ML models |
| 3️⃣ | `3_model_inference.ipynb` | 🧪 **Model Testing** | Performance evaluation with detailed metrics |
| 4️⃣ | `4_deploy.ipynb` | 🚀 **Model Deployment** | Convert to ONNX format for optimization |
| 5️⃣ | `5_live_camera_detection.ipynb` | **🏋️ Live Application** | **Main app - Start here for exercise!** |

## 🎮 User Interface

### Visual Feedback
- **🟢 Green text** → Correct form ✅
- **🔴 Red text** → Form correction needed ⚠️
- **🟠 Orange text** → Technique guidance 💡
- **📊 Rep counters** → Separate tracking for left/right arms
- **💬 Live messages** → "Extend arm fully", "Squeeze bicep", "Good rep!" 🎉

### Controls
- **'q'** → Quit application ❌
- **'r'** → Reset rep counters 🔄

## 🧠 Technical Overview

### AI Architecture
The system uses **MediaPipe** for pose detection and extracts key biomechanical features:

- **🦴 Elbow angles** - Primary indicator for curl range of motion
- **🏃 Shoulder angles** - Stability and postural analysis
- **📏 Arm positioning** - Proper form maintenance
- **⚖️ Bilateral comparison** - Independent left/right arm tracking
- **📐 Geometric features** - Comprehensive movement analysis

### Machine Learning
A **🧮 Naive Bayes model** (with alternative algorithms) analyzes extracted features to classify exercise form accuracy for each arm independently.

## 📁 Project Structure

```
bicep_curl/
├── 📊 data/                    # Training datasets (train.csv, test.csv)
├── 🤖 models/                  # Trained AI models
│   ├── pkl/                   # Scikit-learn models (.pkl files)
│   └── onnx/                  # Optimized deployment models (.onnx files)
├── 📓 notebooks/              # Jupyter notebooks (main workflow)
└── 📖 README.md               # Project documentation
```

## 🏋️ Exercise Guidelines

### ✅ Correct Form
- Stand with feet shoulder-width apart 👣
- Hold weights with arms at sides 💪
- Keep shoulders stable and engaged 🏃
- Curl weight up by bending elbow 🔄
- Squeeze bicep at top (elbow ~75°) 💪
- Lower with control to full extension (elbow ~140°) ⬇️

### ❌ Common Mistakes
- Swinging or using momentum 🚫
- Insufficient arm extension 🚫
- Shoulder movement/leaning 🚫
- Rushed repetitions 🚫
- Incomplete range of motion 🚫

## 🔧 Troubleshooting

### Camera Issues
- **📹 No detection:** Check webcam permissions and lighting
- **🖼️ Poor tracking:** Ensure clear background and full upper body visibility
- **👤 Position:** Stand facing camera with arms clearly visible

### Model Performance
- **🤖 Retraining:** Run notebook 2 with adjusted parameters
- **⚙️ Algorithm:** Try different models (NB, RF, SVM)
- **🔄 Data:** Increase training data for better generalization

### Environment Setup
- **📦 Dependencies:** Verify all packages are correctly installed
- **🔐 Permissions:** Ensure camera access is granted
- **💡 Lighting:** Bright, even lighting improves detection accuracy
- **🖥️ Resources:** Monitor CPU usage during real-time processing

## 🚀 Advanced Configuration

### Model Training Process
1. **Data Preprocessing** → Bilateral feature extraction and normalization
2. **Feature Engineering** → Elbow/shoulder angle calculations
3. **Model Training** → Separate models for left/right arms with cross-validation
4. **Model Evaluation** → Performance metrics and confusion matrices
5. **Model Selection** → Best performers converted to ONNX format

### Performance Optimization
- Use ONNX runtime for faster inference
- Adjust MediaPipe detection confidence thresholds
- Optimize video frame processing rate
- Fine-tune rep counting parameters

### Rep Counting Configuration
```python
ELBOW_ANGLE_UP_THRESHOLD = 85      # Curl up position
ELBOW_ANGLE_DOWN_THRESHOLD = 140   # Extended position
SHOULDER_TOLERANCE = 25            # Stability threshold
REP_COOLDOWN_SECONDS = 0.5         # Minimum time between reps
```

---

**🎉 Ready to perfect your bicep curls?** Open `5_live_camera_detection.ipynb` and start training! 💪✨

**❓ Need help?** Check individual notebook comments for detailed explanations. 📚
