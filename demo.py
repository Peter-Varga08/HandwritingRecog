import gradio as gr
import os
from launch import run_pipeline

# Function to process the input image
def process_image(input_image, *args, **kwargs):
    # Replace this with your actual image processing logic
    output_image_path = f"processed_images/{os.path.basename(input_image)}"  # Save processed image in a folder
    # TODO: Create 'run_demo_pipeline' that takes image as input and returns the processed image
    run_demo_pipeline(input_image, results_path="processed_images")

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