import os
import numpy as np
from vision_agent.tools import *
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool

import numpy as np

def detect_humans_in_image(image_path: str):
    """
    Detect all humans in the given image by subdividing the image, detecting 'person'
    in each subdivision, merging overlapping detections, and visualizing the results
    with bounding boxes.

    Parameters:
        image_path (str): The file path or URL of the image to process.

    Returns:
        List[Dict[str, Any]]: A list of final merged detections, each containing
                              'label', 'score', and 'bbox' in normalized coordinates.
    """
    # Load the image
    image = load_image(image_path)
    height, width, _ = image.shape

    # Helper function: subdivide image into 4 overlapping regions
    def subdivide_image(img: np.ndarray):
        h, w, _ = img.shape
        mid_h = h // 2
        mid_w = w // 2
        overlap_h = int(mid_h * 0.1)
        overlap_w = int(mid_w * 0.1)
        top_left = img[:mid_h + overlap_h, :mid_w + overlap_w, :]
        top_right = img[:mid_h + overlap_h, mid_w - overlap_w:, :]
        bottom_left = img[mid_h - overlap_h:, :mid_w + overlap_w, :]
        bottom_right = img[mid_h - overlap_h:, mid_w - overlap_w:, :]
        return [top_left, top_right, bottom_left, bottom_right]

    # Helper function: check if two bounding boxes match via IoU
    def bounding_box_match(b1, b2, iou_threshold=0.1):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        
        if x2 < x1 or y2 < y1:
            return False
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0
        return iou >= iou_threshold

    # Helper function: merge bounding boxes from all subdivisions
    def merge_bounding_box_list(detections):
        merged_detections = []
        for detection in detections:
            matched_idx = None
            for i, existing_det in enumerate(merged_detections):
                if bounding_box_match(detection["bbox"], existing_det["bbox"]):
                    matched_idx = i
                    break
                    
            if matched_idx is not None:
                if detection["score"] > merged_detections[matched_idx]["score"]:
                    merged_detections[matched_idx] = detection
            else:
                merged_detections.append(detection)
        return merged_detections

    # Subdivide the image
    subdivided_images = subdivide_image(image)

    # Detect humans in each subdivision, then adjust bounding boxes
    all_detections = []
    mid_width = width // 2
    mid_height = height // 2
    for i, sub_img in enumerate(subdivided_images):
        sub_h, sub_w = sub_img.shape[:2]
        # Use the human detection tool with prompt "person"
        detections = countgd_object_detection("person", sub_img)

        # Convert normalized -> unnormalized -> apply offset -> normalize
        offset_x = (i % 2) * (mid_width - int(mid_width * 0.1))
        offset_y = (i // 2) * (mid_height - int(mid_height * 0.1))
        
        for det in detections:
            x1_sub = det["bbox"][0] * sub_w
            y1_sub = det["bbox"][1] * sub_h
            x2_sub = det["bbox"][2] * sub_w
            y2_sub = det["bbox"][3] * sub_h

            x1_global = (x1_sub + offset_x) / width
            y1_global = (y1_sub + offset_y) / height
            x2_global = (x2_sub + offset_x) / width
            y2_global = (y2_sub + offset_y) / height

            new_det = {
                "label": det["label"],
                "score": det["score"],
                "bbox": [x1_global, y1_global, x2_global, y2_global]
            }
            all_detections.append(new_det)

    # Merge overlapping detections and remove duplicates
    final_dets = merge_bounding_box_list(all_detections)

    # Visualize results
    overlaid = overlay_bounding_boxes(image, final_dets)
    save_image(overlaid, "result.jpg")

    # Return the final detections
    return final_dets

detect_humans_in_image("/content/test.png")