# 💪 Bicep Curl Exercise Correction AI

An AI system that analyzes bicep curl form in real-time using your webcam and provides instant feedback. 🎯

## 🚀 What it does

- **📹 Watches your bicep curls** through your webcam
- **🔍 Analyzes your form** using pose detection
- **⚡ Gives real-time feedback** on correct/incorrect technique
- **🔢 Counts your reps** automatically
- **🤝 Works for both arms** separately

## ⚡ Quick Start

1. **📦 Install requirements:**
   ```bash
   pip install opencv-python mediapipe pandas numpy scikit-learn matplotlib seaborn onnx onnxruntime
   ```

2. **📚 Run the notebooks in order:**
   - `1_exploratory_data_analysis.ipynb` - 📊 Look at the data
   - `2_data_preprocessing_and_model_training.ipynb` - 🤖 Train the AI
   - `3_model_inference.ipynb` - 🧪 Test the AI
   - `4_deploy.ipynb` - 🚀 Prepare for live use
   - `5_live_camera_detection.ipynb` - **🏋️ Start exercising!**

## 🎮 How to use

1. Open `5_live_camera_detection.ipynb` 📂
2. Run all cells ▶️
3. Position yourself in front of your webcam 📸
4. Start doing bicep curls 💪
5. Watch the feedback on screen 👀

**🎮 Controls:**
- Press **'q'** to quit ❌
- Press **'r'** to reset rep counter 🔄

## 👁️ What you'll see

- **🟢 Green text** = Good form ✅
- **🔴 Red text** = Fix your form ⚠️
- **📊 Rep counter** for each arm
- **💬 Real-time feedback** like "Extend arm fully" or "Good rep!" 🎉

## 📁 Project Structure

```
bicep_curl/
├── 📊 data/                    # Training data
├── 🤖 models/                  # Trained AI models
├── 📓 notebooks/              # Jupyter notebooks (run these!)
└── 📖 README.md               # This file
```

## 🧠 The Science

The AI uses **MediaPipe** to track your body joints, then calculates:
- **🦴 Elbow angle** (how bent your arm is)
- **🏃 Shoulder angle** (keeping shoulders stable)

A **🧮 Naive Bayes model** decides if your form is correct based on these angles.

## 📋 Files Explained

| 📓 Notebook | 🎯 What it does |
|----------|-------------|
| `1_exploratory_data_analysis.ipynb` | 📊 Shows you what's in the training data |
| `2_data_preprocessing_and_model_training.ipynb` | 🤖 Trains the AI to recognize good/bad form |
| `3_model_inference.ipynb` | 🧪 Tests how well the AI works |
| `4_deploy.ipynb` | 🚀 Converts AI for faster performance |
| `5_live_camera_detection.ipynb` | **🏋️ The main app - use this one!** |

## 🔧 Troubleshooting

**📹 Camera not working?**
- Check webcam permissions 🔐
- Make sure you're well-lit 💡
- Stand where your full upper body is visible 👤

**🤖 AI not detecting you?**
- Move closer/further from camera 📏
- Ensure good lighting 🌟
- Clear background helps 🖼️

**🔄 Want to retrain?**
- Run notebook 2 again with new data 📈
- Adjust the model settings if needed ⚙️

## 💡 Tips for Best Results

- **💡 Good lighting** is important
- **🖼️ Clear background** helps detection
- **👤 Show your full upper body** in the camera
- **⬇️ Start with arms down** to calibrate

---

**❓ Need help?** Check the individual notebook comments for detailed explanations. 📚

**🎉 Happy training!** 💪✨
