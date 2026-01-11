######
# Copy pasted from
# https://huggingface.co/blog/prithivMLmods/multimodal-ocr-vlms#ii-dotsocr
# and extended for batch processing.
#######
import os
import sys
import random
import uuid
import json
import time
import glob
from threading import Thread
from typing import Iterable
from huggingface_hub import snapshot_download

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import cv2

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TextIteratorStreamer,
)

from transformers.image_utils import load_image

css = """
#main-title h1 {
    font-size: 2.3em !important;
}
#output-title h2 {
    font-size: 2.1em !important;
}
"""

MAX_MAX_NEW_TOKENS = 2048  # Reduced from 4096
DEFAULT_MAX_NEW_TOKENS = 1024  # Reduced from 2048
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Dots.OCR
MODEL_PATH_D = "strangervisionhf/dots.ocr-base-fix"
processor = AutoProcessor.from_pretrained(MODEL_PATH_D, trust_remote_code=True)

# Enable memory optimization
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH_D,
    attn_implementation="flash_attention_2",
    dtype=torch.bfloat16,  # Use dtype instead of deprecated torch_dtype
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    max_memory={0: "4GB"},  # Further reduce GPU memory usage to 4GB
).eval()

@spaces.GPU
def generate_image(text: str, image: Image.Image,
                   max_new_tokens: int, temperature: float, top_p: float,
                   top_k: int, repetition_penalty: float):
    """
    Generates responses using the Dots.OCR model for image input.
    Yields raw text and Markdown-formatted text.
    """
    if image is None:
        yield "Please upload an image.", "Please upload an image."
        return
    
    # Aggressive GPU cache clearing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": text},
        ]
    }]
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[prompt_full],
        images=[image],
        return_tensors="pt",
        padding=True).to(device)

    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer, buffer

def process_directory(directory_path: str, text_query: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int, repetition_penalty: float, output_dir: str, start_page: int, progress=gr.Progress()):
    """
    Process all PNG files in a directory and return results.
    """
    if not directory_path or not os.path.exists(directory_path):
        yield None
        return
    
    # Find all PNG files in the directory
    png_files = glob.glob(os.path.join(directory_path, "*.png"))
    
    if not png_files:
        yield None
        return
    
    # Sort files based on numeric suffix (page_1, page_2, etc.)
    def extract_page_number(filename):
        base = os.path.basename(filename)
        # Look for patterns like page_1, page_2, etc.
        import re
        match = re.search(r'page_(\d+)', base, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # Fallback to natural sorting if no page number found
        return base
    
    png_files = sorted(png_files, key=extract_page_number)
    
    # Validate start_page
    if start_page < 1:
        start_page = 1
    elif start_page > len(png_files):
        yield None
        return
    
    # Adjust for 0-based indexing
    start_index = start_page - 1
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    current_image = None
    
    for i in range(start_index, len(png_files)):
        png_file = png_files[i]
        filename = os.path.basename(png_file)
        current_num = i + 1
        total_num = len(png_files)
        # Check if corresponding markdown file already exists
        if output_dir:
            base_name = os.path.splitext(filename)[0]
            md_filename = f"{base_name}.md"
            md_filepath = os.path.join(output_dir, md_filename)
            
            if os.path.exists(md_filepath):
                progress((i - start_index + 0.5) / (total_num - start_index), f"Skipping {filename} ({current_num}/{total_num}) - already processed")
                continue
        
        progress((i - start_index + 0.5) / (total_num - start_index), f"Processing {filename} ({current_num}/{total_num})...")
        
        try:
            # Load image
            image = Image.open(png_file)
            current_image = image  # Update current image for display
            
            # Process with the model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_query},
                ]
            }]
            prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = processor(
                text=[prompt_full],
                images=[image],
                return_tensors="pt",
                padding=True).to(device)

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )
            
            # Decode the output
            generated_text = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the response part (remove the prompt)
            response = generated_text.split("assistant")[-1].strip()
            if response.startswith(":"):
                response = response[1:].strip()
            
            # Save individual markdown file if output directory is specified
            if output_dir:
                # Create markdown filename based on PNG filename
                base_name = os.path.splitext(filename)[0]
                md_filename = f"{base_name}.md"
                md_filepath = os.path.join(output_dir, md_filename)
                
                # Write markdown content to file
                with open(md_filepath, 'w', encoding='utf-8') as md_file:
                    md_file.write(f"# {filename}\n\n{response}\n")
            
            # Clear GPU cache after each image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            # Continue processing even if there's an error
            pass
        
        # Yield current state for real-time updates
        yield current_image
    
    # Final yield to ensure complete results
    yield current_image

with gr.Blocks() as demo:
    gr.Markdown("# **Dots.OCR Demo**", elem_id="main-title")
    
    with gr.Tab("Single Image"):
        with gr.Row():
            with gr.Column(scale=2):
                image_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                image_upload = gr.Image(type="pil", label="Upload Image", height=290)
                image_submit = gr.Button("Submit", variant="primary")
            
                with gr.Accordion("Advanced options", open=False):
                    max_new_tokens_single = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
                    temperature_single = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.7)
                    top_p_single = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                    top_k_single = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                    repetition_penalty_single = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.1)
                    
            with gr.Column(scale=3):
                    gr.Markdown("## Output", elem_id="output-title")
                    output_single = gr.Textbox(label="Raw Output Stream", interactive=False, lines=11)
                    with gr.Accordion("(Result.md)", open=False):
                        markdown_output_single = gr.Markdown(label="(Result.Md)")

        image_submit.click(
            fn=generate_image,
            inputs=[image_query, image_upload, max_new_tokens_single, temperature_single, top_p_single, top_k_single, repetition_penalty_single],
            outputs=[output_single, markdown_output_single]
        )
    
    with gr.Tab("Batch Directory Processing"):
        with gr.Row():
            with gr.Column(scale=2):
                dir_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...", value="Convert to Markdown")
                directory_input = gr.Textbox(label="Directory Path", placeholder="Enter path to directory containing PNG files...")
                start_page_input = gr.Number(label="Start Page (optional)", value=1, minimum=1, step=1, precision=0, info="Page number to start processing from (1-based)")
                output_dir_input = gr.Textbox(label="Output Directory (optional)", placeholder="Enter path to save markdown files...")
                dir_submit = gr.Button("Process Directory", variant="primary")
                
                with gr.Accordion("Advanced options", open=False):
                    max_new_tokens_batch = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
                    temperature_batch = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.7)
                    top_p_batch = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                    top_k_batch = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                    repetition_penalty_batch = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.1)
                    
            with gr.Column(scale=3):
                    gr.Markdown("## Batch Processing", elem_id="output-title")
                    current_image_display = gr.Image(label="Currently Processing", height=400)

        dir_submit.click(
            fn=process_directory,
            inputs=[directory_input, dir_query, max_new_tokens_batch, temperature_batch, top_p_batch, top_k_batch, repetition_penalty_batch, output_dir_input, start_page_input],
            outputs=[current_image_display]
        )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(mcp_server=True, ssr_mode=False, show_error=True, css=css)
