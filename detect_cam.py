# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv5 detection on camera images from directories listed in a txt file.

Supports multiple camera IDs (05, 06, 07, 08, 09) with naming format:
- camera{cam_id}xxx.jpg or camera_{cam_id}xxx.jpg

Usage:
    $ python detect_cam.py --weights yolov5s.pt --source paths.txt --path-prefix /data/images

    # Save both JSON and TXT results
    $ python detect_cam.py --source paths.txt --output-format both

    # With visualization (saves annotated images organized by folder name/timestamp)
    $ python detect_cam.py --source paths.txt --save-viz --viz-dir runs/detect_cam

    # Test mode (uses sample images for validation)
    $ python detect_cam.py --test-mode --save-viz --viz-dir runs/test_viz

Arguments:
    --source: txt file where each line is a directory path containing images
    --path-prefix: prefix to add to relative paths in the source txt file
    --output-format: 'txt', 'json', or 'both' for saving detection results
    --cam-pattern: regex pattern to match camera image names (default matches camera05-09xxx.jpg)
    --save-viz: save visualization images with bounding boxes
    --viz-dir: directory to save visualization images (organized by folder name)
    --test-mode: run in test mode using sample images
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.dataloaders import IMG_FORMATS
from utils.general import (
    LOGGER,
    Profile,
    check_img_size,
    check_requirements,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
)
from utils.torch_utils import select_device, smart_inference_mode


def get_cam_images(directories, path_prefix="", cam_pattern=r"camera_?(0[5-9]).*\.jpg"):
    """
    Get camera image paths from directories, grouped by directory.

    Args:
        directories: list of directory paths
        path_prefix: prefix to add to relative paths
        cam_pattern: regex pattern to match camera image filenames (must have a group for cam_id)

    Returns:
        dict: {dir_path: [(image_path, cam_id), ...]} grouped by directory
    """
    dir_images = {}
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
        images = []
        for file_path in sorted(dir_path.iterdir()):
            if file_path.is_file():
                # Check if file matches the camera pattern
                match = pattern.match(file_path.name)
                if match:
                    # Check if it's a valid image format
                    if file_path.suffix[1:].lower() in IMG_FORMATS:
                        # Extract cam_id from the match group
                        cam_id = match.group(1) if match.groups() else "unknown"
                        images.append((str(file_path), cam_id))

        if images:
            dir_images[str(dir_path)] = images

    return dir_images


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


def save_grouped_detections_txt(grouped_detections, output_path, names):
    """
    Save grouped detections to txt file, organized by cam_id.

    Args:
        grouped_detections: dict {cam_id: {"image": str, "detections": list}}
        output_path: path to save txt file
        names: dict mapping class ids to class names
    """
    with open(output_path, "w") as f:
        for cam_id in sorted(grouped_detections.keys()):
            data = grouped_detections[cam_id]
            f.write(f"# cam_id: {cam_id}\n")
            f.write(f"# image: {data['image']}\n")
            for det in data["detections"]:
                x1, y1, x2, y2, conf, cls = det
                cls_id = int(cls)
                cls_name = names[cls_id] if cls_id in names else str(cls_id)
                f.write(f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {conf:.4f} {cls_id} {cls_name}\n")
            f.write("\n")


def save_grouped_detections_json(grouped_detections, output_path, names):
    """
    Save grouped detections to json file, organized by cam_id.

    Args:
        grouped_detections: dict {cam_id: {"image": str, "detections": list}}
        output_path: path to save json file
        names: dict mapping class ids to class names
    """
    result = {}

    for cam_id, data in grouped_detections.items():
        cam_result = {
            "image": data["image"],
            "detections": [],
        }

        for det in data["detections"]:
            x1, y1, x2, y2, conf, cls = det
            cls_id = int(cls)
            cls_name = names[cls_id] if cls_id in names else str(cls_id)
            cam_result["detections"].append(
                {
                    "bbox": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
                    "confidence": round(float(conf), 4),
                    "class_id": cls_id,
                    "class_name": cls_name,
                }
            )

        result[cam_id] = cam_result

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


def setup_test_mode():
    """
    Setup test mode by creating a temporary directory with sample images.
    Uses existing sample images from data/images directory.
    Creates a folder structure simulating timestamp-based organization.

    Returns:
        tuple: (test_dir, paths_file) - temporary directory and paths.txt file for testing

    Raises:
        FileNotFoundError: If no sample images are found
    """
    test_dir = tempfile.mkdtemp(prefix="detect_cam_test_")
    # Use a timestamp-like folder name to test folder-based visualization organization
    test_subdir = os.path.join(test_dir, "20240101_120000")
    os.makedirs(test_subdir, exist_ok=True)

    # Copy sample images with new camera naming format (camera{cam_id}xxx.jpg)
    sample_images = [
        ROOT / "data/images/bus.jpg",
        ROOT / "data/images/zidane.jpg",
    ]

    # Use camera naming format with different cam_ids (05-09)
    cam_names = ["camera_05001.jpg", "camera_08001.jpg"]

    copied_count = 0
    for src, cam_name in zip(sample_images, cam_names):
        src_path = Path(src).resolve()
        if src_path.exists():
            dst = os.path.join(test_subdir, cam_name)
            shutil.copy(str(src_path), dst)
            LOGGER.info(f"Test mode: copied {src_path.name} -> {cam_name}")
            copied_count += 1
        else:
            LOGGER.warning(f"Test mode: sample image not found: {src_path}")

    if copied_count == 0:
        shutil.rmtree(test_dir)
        raise FileNotFoundError(f"No sample images found in {ROOT / 'data/images'}. Cannot run test mode.")

    # Create paths.txt
    paths_file = os.path.join(test_dir, "paths.txt")
    with open(paths_file, "w") as f:
        f.write(test_subdir + "\n")

    LOGGER.info(f"Test mode: temporary directory created at {test_dir}")
    LOGGER.info(f"Test mode: to clean up manually, run: rm -rf {test_dir}")

    return test_dir, paths_file


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path
    source=ROOT / "paths.txt",  # txt file with directory paths
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    path_prefix="",  # prefix for relative paths
    cam_pattern=r"camera_?(0[5-9]).*\.jpg",  # camera image pattern with cam_id group
    output_format="json",  # output format: 'txt', 'json', or 'both'
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
    save_viz=False,  # save visualization images
    viz_dir="runs/detect_cam",  # directory for visualization output
    test_mode=False,  # run in test mode with sample images
    line_thickness=3,  # bounding box line thickness
):
    """
    Run YOLOv5 detection on camera images from directories listed in a txt file.

    Args:
        weights: Path to model weights file
        source: Path to txt file containing directory paths (one per line)
        data: Path to dataset yaml file
        path_prefix: Prefix to add to relative paths in source file
        cam_pattern: Regex pattern to match camera image filenames (must include group for cam_id)
        output_format: Format for saving results ('txt', 'json', or 'both')
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
        save_viz: Save visualization images with bounding boxes (organized by folder name)
        viz_dir: Directory to save visualization images
        test_mode: Run in test mode using sample images
        line_thickness: Bounding box line thickness for visualization
    """
    test_cleanup_dir = None

    # Handle test mode
    if test_mode:
        LOGGER.info("Running in TEST MODE - using sample images")
        test_cleanup_dir, source = setup_test_mode()
        LOGGER.info(f"Test mode: created temporary directory {test_cleanup_dir}")

    source = str(source)

    # Setup visualization directory
    viz_save_dir = None
    if save_viz:
        viz_save_dir = increment_path(Path(viz_dir), exist_ok=False)
        viz_save_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Visualization images will be saved to: {viz_save_dir}")

    # Read directory paths from source txt file
    if not os.path.isfile(source):
        raise FileNotFoundError(f"Source file not found: {source}")

    with open(source, "r") as f:
        directories = f.readlines()

    # Get camera images grouped by directory
    dir_images = get_cam_images(directories, path_prefix, cam_pattern)
    if not dir_images:
        LOGGER.warning(f"No camera images found matching pattern '{cam_pattern}' in directories from {source}")
        return

    total_images = sum(len(images) for images in dir_images.values())
    LOGGER.info(f"Found {total_images} camera images in {len(dir_images)} directories to process")

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Warmup
    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))

    seen = 0
    dt = (Profile(device=device), Profile(device=device), Profile(device=device))

    # Process each directory
    for dir_path, images in dir_images.items():
        folder_name = Path(dir_path).name  # Use folder name (timestamp) for organization
        grouped_detections = {}  # {cam_id: {"image": str, "detections": list}}

        # Create visualization subdirectory for this folder if needed
        viz_folder_dir = None
        if save_viz:
            viz_folder_dir = viz_save_dir / folder_name
            viz_folder_dir.mkdir(parents=True, exist_ok=True)

        # Process each image in this directory
        for img_path, cam_id in images:
            # Load image
            im0 = cv2.imread(img_path)
            if im0 is None:
                LOGGER.warning(f"Image not found or cannot be read: {img_path}")
                continue

            # Preprocess
            with dt[0]:
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

                # Create annotator for visualization
                annotator = Annotator(im0.copy(), line_width=line_thickness, example=str(names)) if save_viz else None

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in det:
                        detections.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(conf), int(cls)])

                        # Add box to visualization
                        if save_viz and annotator is not None:
                            c = int(cls)
                            label = f"{names[c]} {conf:.2f}"
                            annotator.box_label(xyxy, label, color=colors(c, True))

                # Store detection results grouped by cam_id
                grouped_detections[cam_id] = {
                    "image": Path(img_path).name,
                    "detections": detections,
                }

                # Save visualization image (organized by folder name)
                if save_viz and annotator is not None:
                    viz_img = annotator.result()
                    img_name = Path(img_path).stem
                    viz_path = str(viz_folder_dir / f"{img_name}_viz.jpg")
                    cv2.imwrite(viz_path, viz_img)
                    LOGGER.info(f"  cam_{cam_id}: {len(detections)} detection(s), viz: {viz_path}")
                else:
                    LOGGER.info(f"  cam_{cam_id}: {len(detections)} detection(s)")

        # Save grouped detection results to single file per directory
        if grouped_detections:
            if output_format in ("json", "both"):
                output_path_json = os.path.join(dir_path, "detections.json")
                save_grouped_detections_json(grouped_detections, output_path_json, names)
                LOGGER.info(f"Saved grouped detections to: {output_path_json}")

            if output_format in ("txt", "both"):
                output_path_txt = os.path.join(dir_path, "detections.txt")
                save_grouped_detections_txt(grouped_detections, output_path_txt, names)
                LOGGER.info(f"Saved grouped detections to: {output_path_txt}")

    # Print results
    t = tuple(x.t / seen * 1e3 if seen else 0 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)

    if save_viz:
        LOGGER.info(f"Detection results saved. Visualizations organized by folder in: {viz_save_dir}")
    else:
        LOGGER.info(f"Results saved to original directories. Processed {seen} images.")

    # Note about test mode temporary directory
    if test_cleanup_dir and os.path.exists(test_cleanup_dir):
        LOGGER.info(f"Test mode: to clean up, run: rm -rf {test_cleanup_dir}")


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
        default=r"camera_?(0[5-9]).*\.jpg",
        help="regex pattern to match camera image filenames (default: camera05-09xxx.jpg or camera_05-09xxx.jpg)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["txt", "json", "both"],
        help="output format for detection results (txt, json, or both)",
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
    # Visualization and test mode options
    parser.add_argument("--save-viz", action="store_true", help="save visualization images organized by folder name")
    parser.add_argument(
        "--viz-dir", type=str, default=str(ROOT / "runs/detect_cam"), help="directory for visualization output"
    )
    parser.add_argument("--test-mode", action="store_true", help="run in test mode with sample images")
    parser.add_argument("--line-thickness", type=int, default=3, help="bounding box line thickness for visualization")
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
