# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv5 detection on camera images (cam08xxx.jpg or cam_08xxx.jpg) from directories listed in a txt file.

Usage:
    $ python detect_cam.py --weights yolov5s.pt --source paths.txt --path-prefix /data/images --output-format json

Arguments:
    --source: txt file where each line is a directory path containing images
    --path-prefix: prefix to add to relative paths in the source txt file
    --output-format: 'txt' or 'json' for saving detection results
    --cam-pattern: regex pattern to match camera image names (default matches cam08xxx.jpg or cam_08xxx.jpg)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, LoadImages
from utils.general import (
    LOGGER,
    Profile,
    check_img_size,
    check_requirements,
    colorstr,
    cv2,
    non_max_suppression,
    print_args,
    scale_boxes,
)
from utils.torch_utils import select_device, smart_inference_mode


def get_cam_images(directories, path_prefix="", cam_pattern=r"cam_?08.*\.jpg"):
    """
    Get camera image paths from directories.

    Args:
        directories: list of directory paths
        path_prefix: prefix to add to relative paths
        cam_pattern: regex pattern to match camera image filenames

    Returns:
        list of (image_path, output_dir) tuples where output_dir is the original directory
    """
    images = []
    pattern = re.compile(cam_pattern, re.IGNORECASE)

    for dir_path in directories:
        dir_path = dir_path.strip()
        if not dir_path:
            continue

        # Handle relative paths with prefix
        if not os.path.isabs(dir_path) and path_prefix:
            dir_path = os.path.join(path_prefix, dir_path)

        dir_path = Path(dir_path).resolve()

        if not dir_path.exists():
            LOGGER.warning(f"Directory not found: {dir_path}")
            continue

        if not dir_path.is_dir():
            LOGGER.warning(f"Not a directory: {dir_path}")
            continue

        # Find matching images in this directory
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                # Check if file matches the camera pattern
                if pattern.match(file_path.name):
                    # Check if it's a valid image format
                    if file_path.suffix[1:].lower() in IMG_FORMATS:
                        images.append((str(file_path), str(dir_path)))

    return images


def save_detections_txt(detections, output_path, names):
    """
    Save detections to txt file in format: x1 y1 x2 y2 confidence class_id class_name

    Args:
        detections: list of [x1, y1, x2, y2, conf, cls] for each detection
        output_path: path to save txt file
        names: dict mapping class ids to class names
    """
    with open(output_path, "w") as f:
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cls_id = int(cls)
            cls_name = names[cls_id] if cls_id in names else str(cls_id)
            f.write(f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {conf:.4f} {cls_id} {cls_name}\n")


def save_detections_json(detections, output_path, names, image_name):
    """
    Save detections to json file.

    Args:
        detections: list of [x1, y1, x2, y2, conf, cls] for each detection
        output_path: path to save json file
        names: dict mapping class ids to class names
        image_name: name of the source image
    """
    result = {
        "image": image_name,
        "detections": [],
    }

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls_id = int(cls)
        cls_name = names[cls_id] if cls_id in names else str(cls_id)
        result["detections"].append(
            {
                "bbox": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
                "confidence": round(float(conf), 4),
                "class_id": cls_id,
                "class_name": cls_name,
            }
        )

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path
    source=ROOT / "paths.txt",  # txt file with directory paths
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    path_prefix="",  # prefix for relative paths
    cam_pattern=r"cam_?08.*\.jpg",  # camera image pattern
    output_format="json",  # output format: 'txt' or 'json'
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
):
    """
    Run YOLOv5 detection on camera images from directories listed in a txt file.

    Args:
        weights: Path to model weights file
        source: Path to txt file containing directory paths (one per line)
        data: Path to dataset yaml file
        path_prefix: Prefix to add to relative paths in source file
        cam_pattern: Regex pattern to match camera image filenames
        output_format: Format for saving results ('txt' or 'json')
        imgsz: Inference image size (height, width)
        conf_thres: Confidence threshold for detections
        iou_thres: IOU threshold for NMS
        max_det: Maximum detections per image
        device: CUDA device or 'cpu'
        classes: Filter by class indices
        agnostic_nms: Class-agnostic NMS
        augment: Augmented inference
        half: FP16 half-precision inference
        dnn: Use OpenCV DNN for ONNX inference
    """
    source = str(source)

    # Read directory paths from source txt file
    if not os.path.isfile(source):
        raise FileNotFoundError(f"Source file not found: {source}")

    with open(source, "r") as f:
        directories = f.readlines()

    # Get camera images
    cam_images = get_cam_images(directories, path_prefix, cam_pattern)
    if not cam_images:
        LOGGER.warning(f"No camera images found matching pattern '{cam_pattern}' in directories from {source}")
        return

    LOGGER.info(f"Found {len(cam_images)} camera images to process")

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Warmup
    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))

    seen = 0
    dt = (Profile(device=device), Profile(device=device), Profile(device=device))

    for img_path, output_dir in cam_images:
        # Load image
        im0 = cv2.imread(img_path)
        if im0 is None:
            LOGGER.warning(f"Image not found or cannot be read: {img_path}")
            continue

        # Preprocess
        with dt[0]:
            from utils.augmentations import letterbox

            im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = im.astype("float32") / 255.0  # 0-255 to 0.0-1.0
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=augment)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process detections
        for det in pred:
            seen += 1
            detections = []

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det:
                    detections.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(conf), int(cls)])

            # Save results to original directory
            img_name = Path(img_path).stem
            if output_format == "json":
                output_path = os.path.join(output_dir, f"{img_name}_det.json")
                save_detections_json(detections, output_path, names, Path(img_path).name)
            else:  # txt format
                output_path = os.path.join(output_dir, f"{img_name}_det.txt")
                save_detections_txt(detections, output_path, names)

            n_det = len(detections)
            LOGGER.info(f"{img_path}: {n_det} detection{'s' * (n_det != 1)} -> {output_path}")

    # Print results
    t = tuple(x.t / seen * 1e3 if seen else 0 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    LOGGER.info(f"Results saved to original directories. Processed {seen} images.")


def parse_opt():
    """Parse command-line arguments for YOLOv5 camera image detection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path")
    parser.add_argument("--source", type=str, default=ROOT / "paths.txt", help="txt file with directory paths")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--path-prefix", type=str, default="", help="prefix for relative paths in source file")
    parser.add_argument(
        "--cam-pattern",
        type=str,
        default=r"cam_?08.*\.jpg",
        help="regex pattern to match camera image filenames (default: cam08xxx.jpg or cam_08xxx.jpg)",
    )
    parser.add_argument(
        "--output-format", type=str, default="json", choices=["txt", "json"], help="output format for detection results"
    )
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Execute YOLOv5 camera image detection based on command-line arguments."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
