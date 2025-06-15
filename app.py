import streamlit as st
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import cv2
from PIL import Image

# COCO class names
COCO_INSTANCE_CATEGORY_NAMES = [
    '_background_', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Load model
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

st.title("🎯 Image Segmentation App")
st.caption("Using Mask R-CNN with PyTorch")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        score_threshold = st.slider("Score Threshold", 0.0, 1.0, 0.7, 0.05)

        if st.button("Run Detection"):
            # Convert image
            img_tensor = F.to_tensor(image).unsqueeze(0)

            # Run detection
            with torch.no_grad():
                prediction = model(img_tensor)[0]

            # Convert image for overlay
            image_np = np.array(image)

            # Overlay masks
            for i in range(len(prediction["boxes"])):
                score = prediction["scores"][i].item()
                if score < score_threshold:
                    continue

                box = prediction["boxes"][i].cpu().numpy().astype(int)
                label = prediction["labels"][i].item()
                mask = prediction["masks"][i, 0].cpu().numpy()
                mask_binary = mask > 0.5

                color = np.random.randint(0, 256, (3,), dtype=np.uint8)
                image_np[mask_binary] = (0.5 * image_np[mask_binary] + 0.5 * color).astype(np.uint8)

                cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), color.tolist(), 2)
                label_name = COCO_INSTANCE_CATEGORY_NAMES[label]
                cv2.putText(image_np, f"{label_name} {score:.2f}", (box[0], box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

            st.image(image_np, caption="Detected Image", use_container_width=True)
    except Exception as e:
        st.error(f"Error processing the image: {e}")