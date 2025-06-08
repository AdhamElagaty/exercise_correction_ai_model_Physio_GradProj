# 🏃‍♂️ Plank Exercise Correction AI

An AI-powered system that analyzes plank form in real-time using computer vision and provides instant feedback for perfect technique. 🎯

## ✨ Features

- **📹 Real-time pose analysis** through webcam
- **🔍 Intelligent form correction** using pose detection
- **⚡ Instant feedback** on technique accuracy
- **🎯 Three-class classification** (Correct, High Hips, Low Hips)
- **📊 Confidence scoring** for form assessment
- **🔧 Smoothing & debouncing** for stable feedback
- **📈 Visual progress indicators** and feedback display

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
| 2️⃣ | `2_data_preprocessing_and_model_training.ipynb` | 🤖 **Model Training** | Feature extraction, pipeline creation, ML training |
| 3️⃣ | `3_model_inference.ipynb` | 🧪 **Model Testing** | Performance evaluation with detailed metrics |
| 4️⃣ | `4_deploy.ipynb` | 🚀 **Model Deployment** | Convert to ONNX format for optimization |
| 5️⃣ | `5_live_camera_detection.ipynb` | **🏋️ Live Application** | **Main app - Start here for exercise!** |

## 🎮 User Interface

### Visual Feedback
- **🟢 Green text** → Correct form ✅
- **🔴 Red text** → High hips detected ⚠️
- **🟠 Orange text** → Low hips detected ⚠️
- **🟡 Yellow text** → Analyzing/Adjusting 🔄
- **📊 Confidence score** → Model certainty display
- **🎯 Real-time status** → Live form assessment

### Controls
- **'q'** → Quit application ❌

## 🧠 Technical Overview

### AI Architecture
The system uses **MediaPipe** for pose detection and extracts key biomechanical features:

- **💪 Elbow angles** - Arm positioning and stability
- **🏃 Shoulder angles** - Upper body alignment  
- **🦴 Hip angles** - Core engagement assessment
- **🦵 Knee angles** - Lower body positioning
- **📏 Body alignment** - Straight line maintenance
- **📐 Hip deviation** - Distance from optimal position
- **📊 Geometric features** - Comprehensive form analysis (20 features total)

### Machine Learning Pipeline
A **🌲 Random Forest model** (with alternative algorithms) processes extracted features through:
- **🔧 SimpleImputer** - Handles missing landmark data
- **📊 StandardScaler** - Feature normalization
- **🤖 Classification Model** - Form accuracy prediction

Available models: **Logistic Regression**, **KNN**, **Decision Tree**, **Random Forest**, **XGBoost**

## 📁 Project Structure

```
plank/
├── 📊 data/                    # Training datasets (train.csv, test.csv)
├── 🤖 models/                  # Trained AI models
│   ├── pkl/                   # Scikit-learn models (.pkl files)
│   └── onnx/                  # Optimized deployment models (.onnx files)
├── 📓 notebooks/              # Jupyter notebooks (main workflow)
└── 📖 README.md               # Project documentation
```

## 🏋️ Exercise Guidelines

### ✅ Correct Plank Form
- Start in push-up position 🤲
- Forearms on ground, elbows under shoulders 💪
- Body forms straight line: head → shoulders → hips → ankles 📏
- Engage core muscles 🔥
- Keep hips level (not too high or low) ⚖️
- Maintain neutral spine 🦴
- Breathe steadily 🫁

### ❌ Common Mistakes Detected
- **🔴 High Hips:** Hips raised too high (creates inverted V) 🚫
- **🟠 Low Hips:** Hips sagging too low (back arches) 🚫  
- **⚠️ Poor alignment:** Body not maintaining straight line 🚫

## 🔧 Configuration Options

### Detection Settings
```python
VISIBILITY_THRESHOLD = 0.5        # Landmark visibility requirement
SMOOTHING_WINDOW_SIZE = 7         # Prediction smoothing
STATUS_HISTORY_LENGTH = 10        # Status consistency checking
STATUS_COOLDOWN_SEC = 0.5         # Minimum time between status changes
CONFIDENCE_THRESHOLD_CORRECT = 0.65   # Confidence for "correct" display
CONFIDENCE_THRESHOLD_WRONG = 0.50     # Confidence for error display
```

### Model Selection
Change `MODEL_KEY` in notebooks to use different algorithms:
- **'RF'** - Random Forest (default, best performance)
- **'XGB'** - XGBoost (gradient boosting)
- **'LR'** - Logistic Regression (linear model)
- **'KNN'** - K-Nearest Neighbors (similarity-based)
- **'DT'** - Decision Tree (rule-based)

## 🔧 Troubleshooting

### Camera Issues
- **📹 No detection:** Check webcam permissions and lighting
- **🖼️ Poor tracking:** Ensure clear background and full body visibility
- **👤 Position:** Position sideways to camera for optimal profile view
- **💡 Lighting:** Bright, even lighting improves detection accuracy

### Model Performance
- **🤖 Retraining:** Run notebook 2 with different `CHOSEN_MODEL_KEY`
- **⚙️ Algorithm:** Try different models (RF recommended for best results)
- **🔄 Data:** Check for sufficient training data quality
- **📊 Thresholds:** Adjust confidence thresholds for sensitivity

### Environment Setup
- **📦 Dependencies:** Verify all packages are correctly installed
- **🔐 Permissions:** Ensure camera access is granted
- **🖥️ Resources:** Monitor CPU usage during live detection
- **🚀 ONNX:** Use ONNX models for faster inference

## 🚀 Advanced Configuration

### Feature Engineering
The system extracts **20 geometric features** from pose landmarks:
- Joint angles (elbows, shoulders, hips, knees)
- Body alignment angles
- Hip deviation measurements
- Vertical position differences
- Segment length calculations

### Performance Optimization
- **ONNX Runtime** for 3x faster inference
- **Prediction smoothing** reduces jittery feedback
- **Status debouncing** prevents rapid status changes
- **Visibility filtering** ensures reliable landmark data

### Training Process
1. **📊 EDA** → Understand data distribution and quality
2. **🔧 Preprocessing** → Feature extraction with GeometryUtils
3. **🤖 Pipeline Creation** → Imputation + Scaling + Model
4. **📈 Training** → Multiple algorithms with balanced classes
5. **📊 Evaluation** → Classification reports and confusion matrices
6. **🚀 Deployment** → ONNX conversion for production use

---

**🎉 Ready to perfect your planks?** Open `5_live_camera_detection.ipynb` and start training! 🏃‍♂️✨

**❓ Need help?** Check individual notebook comments for detailed explanations. 📚

---

*Built with MediaPipe, scikit-learn, and ONNX Runtime for real-time exercise form correction* 🤖
