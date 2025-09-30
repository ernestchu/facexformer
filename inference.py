"""
FaceXFormer: A Unified Transformer for Facial Analysis

Supported tasks:
- Face Parsing (Task 0)
- Face Landmarks Detection (Task 1)
- Face Headpose Estimation (Task 2)
- Face Attributes Recognition (Task 3)
- Face Age/Gender/Race Estimation (Task 4)
- Face Landmarks Visibility Prediction (Task 5)
"""
import os
import sys
import argparse
from pathlib import Path
from math import cos, sin

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import InterpolationMode
from PIL import Image
from huggingface_hub import hf_hub_download

from network import FaceXFormer
from facenet_pytorch import MTCNN


# Constants
TASK_MAPPING = {
    "parsing": 0,
    "landmarks": 1,
    "headpose": 2,
    "attributes": 3,
    "age_gender_race": 4,
    "visibility": 5
}

DEFAULT_MODEL_REPO = "kartiknarayan/facexformer"
DEFAULT_MODEL_FILENAME = "ckpts/model.pt"


def download_model(model_path=None, repo_id=DEFAULT_MODEL_REPO, filename=DEFAULT_MODEL_FILENAME):
    """
    Download model weights from HuggingFace if not already present.
    
    Args:
        model_path: Path to the model file. If None or doesn't exist, downloads from HF
        repo_id: HuggingFace repository ID
        filename: Filename in the repository
        
    Returns:
        Path to the model file
    """
    if model_path and Path(model_path).exists():
        print(f"Using model from: {model_path}")
        return model_path
    
    print(f"Downloading model from HuggingFace ({repo_id})...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir="./",
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded successfully to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)


def visualize_mask(image_tensor, mask):
    image = image_tensor.numpy().transpose(1, 2, 0) * 255 
    image = image.astype(np.uint8)
    
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_mapping = np.array([
        [0, 0, 0],
        [0, 153, 255],
        [102, 255, 153],
        [0, 204, 153],
        [255, 255, 102],
        [255, 255, 204],
        [255, 153, 0],
        [255, 102, 255],
        [102, 0, 51],
        [255, 204, 255],
        [255, 0, 102]
    ])
    
    for index, color in enumerate(color_mapping):
        color_mask[mask == index] = color

    overlayed_image = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)

    return overlayed_image, image, color_mask

def visualize_landmarks(im, landmarks, color, thickness=3, eye_radius=0):
    im = im.permute(1, 2, 0).numpy()
    im = (im * 255).astype(np.uint8)
    im = np.ascontiguousarray(im)
    landmarks = landmarks.squeeze().numpy().astype(np.int32)
    for (x, y) in landmarks:
        cv2.circle(im, (x,y), eye_radius, color, thickness)
    return im

def visualize_head_pose(img, euler, tdx=None, tdy=None, size = 100):
    pitch, yaw, roll = euler[0], euler[1], euler[2]

    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = np.ascontiguousarray(img)

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,255,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(255,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,255),2)
    return img

def denorm_points(points, h, w, align_corners=False):
    if align_corners:
        denorm_points = (points + 1) / 2 * torch.tensor([w - 1, h - 1], dtype=torch.float32).to(points).view(1, 1, 2)
    else:
        denorm_points = ((points + 1) * torch.tensor([w, h], dtype=torch.float32).to(points).view(1, 1, 2) - 1) / 2

    return denorm_points



def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    tensor = tensor * std + mean 
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def adjust_bbox(x_min, y_min, x_max, y_max, image_width, image_height, margin_percentage=50):
    width = x_max - x_min
    height = y_max - y_min
    
    increase_width = width * (margin_percentage / 100.0) / 2
    increase_height = height * (margin_percentage / 100.0) / 2
    
    x_min_adjusted = max(0, x_min - increase_width)
    y_min_adjusted = max(0, y_min - increase_height)
    x_max_adjusted = min(image_width, x_max + increase_width)
    y_max_adjusted = min(image_height, y_max + increase_height)
    
    return x_min_adjusted, y_min_adjusted, x_max_adjusted, y_max_adjusted


def run_inference(args):
    """Run inference on a single image for the specified task."""
    # Setup device
    device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    print(f"Using device: {device}")
    
    # Download/load model
    model_path = download_model(args.model_path)
    
    # Load model
    print("Loading FaceXFormer model...")
    model = FaceXFormer().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict_backbone'])
    model.eval()

    # Setup image transforms
    transforms_image = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Detect face
    print("Detecting face...")
    mtcnn = MTCNN(keep_all=True, device=device)
    image = Image.open(args.image_path).convert('RGB')
    width, height = image.size
    boxes, probs = mtcnn.detect(image)
    
    if boxes is None or len(boxes) == 0:
        print("No face detected in the image!")
        sys.exit(1)
    
    # Adjust bbox and crop face
    x_min, y_min, x_max, y_max = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]
    x_min, y_min, x_max, y_max = adjust_bbox(x_min, y_min, x_max, y_max, width, height)
    image = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
    image = transforms_image(image)

    # Prepare task tensor
    task_id = TASK_MAPPING[args.task]
    task = torch.tensor([task_id])
    
    # Prepare data
    data = {
        'image': image,
        'label': {
            "segmentation": torch.zeros([224, 224]),
            "lnm_seg": torch.zeros([5, 2]),
            "landmark": torch.zeros([68, 2]),
            "headpose": torch.zeros([3]),
            "attribute": torch.zeros([40]),
            "a_g_e": torch.zeros([3]),
            'visibility': torch.zeros([29])
        },
        'task': task
    }
    
    images, labels, tasks = data["image"], data["label"], data["task"]
    images = images.unsqueeze(0).to(device=device)
    for k in labels.keys():
        labels[k] = labels[k].unsqueeze(0).to(device=device)
    tasks = tasks.to(device=device)

    # Run inference
    print(f"Running inference for task: {args.task}")
    with torch.no_grad():
        landmark_output, headpose_output, attribute_output, visibility_output, \
        age_output, gender_output, race_output, seg_output = model(images, labels, tasks)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process outputs based on task
    if task_id == 0:  # parsing
        preds = seg_output.softmax(dim=1)
        mask = torch.argmax(preds, dim=1)
        pred_mask = mask[0].detach().cpu().numpy()
        save_path = os.path.join(args.output_dir, "parsing.png")
        cv2.imwrite(save_path, pred_mask)
        
        mask_viz, face, color_mask = visualize_mask(unnormalize(images[0].detach().cpu()), pred_mask)
        save_path_viz = os.path.join(args.output_dir, "parsing_visualization.png")
        cv2.imwrite(save_path_viz, mask_viz[:, :, ::-1])
        print(f"Saved parsing results to: {args.output_dir}")
        
    elif task_id == 1:  # landmarks
        image_viz = unnormalize(images[0].detach().cpu())
        denorm_landmarks = denorm_points(landmark_output.view(-1, 68, 2)[0], 224, 224)
        im = visualize_landmarks(image_viz, denorm_landmarks.detach().cpu(), (255, 255, 0))
        save_path_viz = os.path.join(args.output_dir, "landmarks.png")
        save_path = os.path.join(args.output_dir, "landmarks.txt")
        cv2.imwrite(save_path_viz, im[:, :, ::-1])
        with open(save_path, 'w') as file:
            for landmark in denorm_landmarks[0]:
                x, y = landmark[0], landmark[1]
                file.write(f"{x.item()} {y.item()}\n")
        print(f"Saved landmarks results to: {args.output_dir}")
        
    elif task_id == 2:  # headpose
        image_viz = unnormalize(images[0].detach().cpu())
        im = visualize_head_pose(image_viz, headpose_output[0])
        save_path_viz = os.path.join(args.output_dir, "headpose.png")
        save_path = os.path.join(args.output_dir, "headpose.txt")
        cv2.imwrite(save_path_viz, im[:, :, ::-1])
        with open(save_path, 'w') as file:
            file.write(f"Pitch: {headpose_output[0][0].item() * 180 / np.pi}\n")
            file.write(f"Yaw: {headpose_output[0][1].item() * 180 / np.pi}\n")
            file.write(f"Roll: {headpose_output[0][2].item() * 180 / np.pi}")
        print(f"Saved headpose results to: {args.output_dir}")
        
    elif task_id == 3:  # attributes
        probs = torch.sigmoid(attribute_output[0])
        preds = (probs >= 0.5).float()
        pred = preds.tolist()
        pred_str = [str(int(b)) for b in pred]
        joined_pred = " ".join(pred_str)
        save_path = os.path.join(args.output_dir, "attributes.txt")
        with open(save_path, 'w') as file:
            file.write(joined_pred)
        print(f"Saved attributes results to: {args.output_dir}")
        
    elif task_id == 4:  # age_gender_race
        age_preds = torch.argmax(age_output, dim=1)[0]
        gender_preds = torch.argmax(gender_output, dim=1)[0]
        race_preds = torch.argmax(race_output, dim=1)[0]
        save_path = os.path.join(args.output_dir, "age_gender_race.txt")
        with open(save_path, 'w') as file:
            file.write(f"Age: {age_preds.item()}\n")
            file.write(f"Gender: {gender_preds.item()}\n")
            file.write(f"Race: {race_preds.item()}")
        print(f"Saved age/gender/race results to: {args.output_dir}")
        
    elif task_id == 5:  # visibility
        probs = torch.sigmoid(visibility_output[0])
        preds = (probs >= 0.5).float()
        pred = preds.tolist()
        pred_str = [str(int(b)) for b in pred]
        joined_pred = " ".join(pred_str)
        save_path = os.path.join(args.output_dir, "visibility.txt")
        with open(save_path, 'w') as file:
            file.write(joined_pred)
        print(f"Saved visibility results to: {args.output_dir}")
    
    # Save cropped face
    image_face = unnormalize(images[0].detach().cpu())
    image_face = image_face.permute(1, 2, 0).numpy()
    image_face = (image_face * 255).astype(np.uint8)
    save_path = os.path.join(args.output_dir, "face.png")
    cv2.imwrite(save_path, image_face[:, :, ::-1])
    
    print("Inference completed successfully!")


def main():
    """Main entry point for FaceXFormer inference."""
    parser = argparse.ArgumentParser(
        description="FaceXFormer: A Unified Transformer for Facial Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run face parsing
  python inference.py --image image.jpg --task parsing
  
  # Run landmark detection with custom output directory
  python inference.py --image image.jpg --task landmarks --output-dir ./my_results
  
  # Use specific model path
  python inference.py --image image.jpg --task headpose --model-path ./ckpts/model.pt
  
Supported tasks:
  parsing          - Face parsing/segmentation
  landmarks        - Face landmark detection (68 points)
  headpose         - Head pose estimation (pitch, yaw, roll)
  attributes       - Face attributes recognition (40 attributes)
  age_gender_race  - Age, gender, and race estimation
  visibility       - Face landmark visibility prediction
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--image",
        "--image-path",
        "--image_path",
        dest="image_path",
        type=str,
        required=True,
        help="Path to the input image"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_MAPPING.keys()),
        help="Task to perform"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model-path",
        "--model_path",
        dest="model_path",
        type=str,
        default=None,
        help="Path to model weights. If not provided, will download from HuggingFace"
    )
    
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        "--results-path",
        "--results_path",
        dest="output_dir",
        type=str,
        default="./results",
        help="Directory to save results (default: ./results)"
    )
    
    parser.add_argument(
        "--device",
        "--gpu-num",
        "--gpu_num",
        dest="device",
        type=int,
        default=0,
        help="GPU device number. Use -1 for CPU (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Validate input image exists
    if not Path(args.image_path).exists():
        print(f"Error: Input image not found: {args.image_path}")
        sys.exit(1)
    
    # Run inference
    run_inference(args)


if __name__ == "__main__":
    main()
