# Vehicle Detection for Autonomous Vehicles

A computer vision application that detects vehicles in video footage using YOLOv8, optimized for autonomous vehicle scenarios.

---

## 📷 Overview

This application processes video files to identify and track vehicles using a YOLOv8 model trained on the KITTI dataset. The model is optimized for autonomous driving scenarios and deployed using ONNX for efficient inference.

---

## 🚀 Features

- Real-time vehicle detection in video files  
- Interactive web interface powered by Streamlit  
- Support for various video formats  
- Bounding box visualization with confidence scores  
- Downloadable output video with detection overlays  
- ONNX runtime optimization for faster inference  

---

## 🧰 Requirements

- Python 3.10+  
- Dependencies listed in `requirements.txt`  
- Pre-trained ONNX model file (`model/best.onnx`)  

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/object-detection-for-autonomous-vehicles.git
   cd object-detection-for-autonomous-vehicles
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the model file is in place:**
   ```
   model/best.onnx
   ```

---

## ▶️ Usage

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically `http://localhost:8501`)

3. Upload a video file using the interface

4. View real-time detection results and download the processed video

---

## 📁 Project Structure

```
object-detection-for-autonomous-vehicles/
├── app.py                    # Main Streamlit application
├── model/
│   └── best.onnx             # YOLOv8 ONNX model file
├── notebooks/                # Jupyter notebooks for model training and testing
│   ├── model_training.ipynb
│   └── inference_test.ipynb
├── src/
│   └── video_utils.py        # Helper functions for video processing
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🛠️ Technical Details

- **Frontend:** Streamlit provides an interactive web interface  
- **Computer Vision:** OpenCV handles video processing and frame manipulation  
- **ML Framework:** ONNX Runtime for optimized model inference  
- **Model:** YOLOv8 architecture trained on the KITTI dataset  
- **Video Processing:** Custom pipeline for efficient frame-by-frame processing  

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [KITTI dataset](http://www.cvlibs.net/datasets/kitti/) for providing training data  
- [Ultralytics](https://github.com/ultralytics) for the YOLOv8 implementation  
- [Streamlit](https://streamlit.io/) for the web interface framework  
