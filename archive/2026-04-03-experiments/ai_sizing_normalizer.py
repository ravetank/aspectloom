import os
import json
import base64
import requests
from PIL import Image, ImageOps

# --- CONFIGURATION ---
INPUT_DIR = r"C:\Path\To\Your\Images"
OUTPUT_DIR = r"C:\Path\To\Your\Output"
COMFY_URL = "http://127.0.0.1:8188/prompt"
WORKFLOW_PATH = "outpainting_workflow_v2.json"
TARGET_RES = (3840, 2160) # 4K
PADDING_COLOR = (0, 0, 255) # Pure Blue for the Mask

def process_images():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(WORKFLOW_PATH, 'r', encoding='utf-8') as f:
        workflow = json.load(f)

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing: {filename}")

            # 1. Prepare 4K Canvas
            img = Image.open(os.path.join(INPUT_DIR, filename))
            canvas = Image.new('RGB', TARGET_RES, PADDING_COLOR)

            # Center the image
            offset = ((TARGET_RES[0] - img.size[0]) // 2, (TARGET_RES[1] - img.size[1]) // 2)
            canvas.paste(img, offset)

            # Save temporary canvas for ComfyUI to pick up
            temp_path = os.path.join(os.getcwd(), "input_temp.png")
            canvas.save(temp_path)

            # 2. Update Workflow JSON
            # Node 8 is your LoadImage node
            workflow["8"]["inputs"]["image"] = temp_path

            # 3. Send to ComfyUI
            p = {"prompt": workflow}
            response = requests.post(COMFY_URL, json=p)

            if response.status_code == 200:
                print(f"Successfully queued {filename}")
            else:
                print(f"Error for {filename}: {response.text}")

if __name__ == "__main__":
    process_images()
