from pathlib import Path

import gradio as gr
import os
from PIL import Image
from launch import run_pipeline


# Function to process the input image
def process_image(image: Image, *args, **kwargs):
    run_pipeline(image, results_path="gradio_results")

# Gradio Interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Input Image", mirror_webcam=False),
    outputs=gr.Image(type="pil", label="Processed Image"),
    examples=[
        "data/image-data/binary/P21-Fg006-R-C01-R01-binarized.jpg",
        "data/image-data/binary/P22-Fg008-R-C01-R01-binarized.jpg",
        "data/image-data/binary/P172-Fg001-R-C01-R01-binarized.jpg"
    ],
)

# Launch the Gradio app
iface.launch()