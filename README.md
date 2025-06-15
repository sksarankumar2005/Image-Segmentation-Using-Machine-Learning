# Image Segmentation using Machine Learning

An interactive web application for instance segmentation of images using Mask R-CNN (PyTorch) and Streamlit. Upload your own images and see pixel-level object segmentation in real time!

---

## Features

- Instance segmentation for multiple object types (person, car, motorcycle, etc.)
- Pretrained Mask R-CNN (ResNet-50 FPN) on COCO dataset
- Easy image upload and threshold adjustment via Streamlit web app
- Visual output with color-coded masks, bounding boxes, and class labels

---

## Demo

Below is a sample input image and the corresponding output from the application:

**Input Image:**

![Input Image](input_sample.png)

**Segmented Output:**

![Sample Output](sample_output.png)


## Installation

1. **Clone this repository:**
    ```
    git clone https://github.com/yourusername/image-segmentation-ml.git
    cd image-segmentation-ml
    ```

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```
    Main dependencies:
    - streamlit
    - torch
    - torchvision
    - numpy
    - opencv-python
    - pillow

3. **Run the application:**
    ```
    streamlit run app.py
    ```

---

## Usage

1. Launch the application using the appropriate command (e.g., `streamlit run app.py`).
2. Open the provided application link in your web browser (the link will be displayed in the terminal or notebook output).
3. Upload an image file (JPG, JPEG, or PNG) using the interface.
4. Adjust the score threshold slider to filter out low-confidence detections as needed.
5. Click **Run Detection** to perform segmentation and labeling of objects in your image.
6. View the output image with color-coded masks, bounding boxes, and class labels directly in the app.

> **Note:**  
> - If running locally, the link is typically `http://localhost:8501`.  
> - If running in Google Colab or a remote environment, use the public URL provided by your tunneling tool (e.g., ngrok).

---

## How It Works

- The app loads a pretrained Mask R-CNN model (`maskrcnn_resnet50_fpn`) from PyTorch, trained on the COCO dataset.
- When you upload an image, it is preprocessed and passed through the model.
- The model outputs segmentation masks, bounding boxes, class labels, and confidence scores for each detected object.
- The app overlays this information on your image and displays it in the browser.

---

## System Requirements

- **Python:** 3.7 or higher
- **Hardware:** 8GB+ RAM recommended. NVIDIA GPU with CUDA support is optional but speeds up inference.
- **OS:** Windows, Linux, or macOS

---


## References

- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [PyTorch Documentation](https://pytorch.org/)
- [COCO Dataset](https://cocodataset.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

