import os
import numpy as np
from vision_agent.tools import *
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool

def detect_humans_in_media(media_path: str, is_video: bool = False) -> List[List[Dict[str, Any]]]:
    """
    Detects all humans in either an image or a video.
    Subdivides large frames if necessary to handle small objects.
    Overlapping bounding boxes are merged.
    Saves visual result as an image or a video, then returns all bounding boxes.
    """
    # Load media
    if is_video:
        try:
            extracted = extract_frames_and_timestamps(media_path)
            frames = [item["frame"] for item in extracted]
            if not frames:  # If no frames extracted, fallback to image mode
                frame = load_image(media_path)
                frames = [frame]
        except:
            # Fallback to image mode if video extraction fails
            frame = load_image(media_path)
            frames = [frame]
    else:
        # Single image input
        frame = load_image(media_path)
        frames = [frame]

    results_per_frame = []
    processed_frames = []  # New list to store frames with bounding boxes

    for idx, frame in enumerate(frames):
        height, width, _ = frame.shape
        subdivided = subdivide_image(frame)

        # Detect humans in each subdivided portion
        partial_detections = []
        for i, sub_image in enumerate(subdivided):
            detections = countgd_object_detection("person", sub_image)
            sub_h, sub_w, _ = sub_image.shape
            # Convert bounding boxes back to the original frame
            offset_x = i % 2 * (width // 2 - int((width // 2) * 0.1))
            offset_y = i // 2 * (height // 2 - int((height // 2) * 0.1))
            # Expand normalized to unnormalized + offset
            for det in detections:
                # unnormalized coords
                x_min = det["bbox"][0] * sub_w + offset_x
                y_min = det["bbox"][1] * sub_h + offset_y
                x_max = det["bbox"][2] * sub_w + offset_x
                y_max = det["bbox"][3] * sub_h + offset_y
                # then normalize to the full frame dimension
                partial_detections.append({
                    "label": det["label"],
                    "score": det["score"],
                    "bbox": [
                        x_min / width,
                        y_min / height,
                        x_max / width,
                        y_max / height
                    ]
                })

        # Merge overlapping detections
        merged = []
        for det in partial_detections:
            found_match = None
            for j, existing in enumerate(merged):
                if bounding_box_match(det["bbox"], existing["bbox"], 0.1):
                    found_match = j
                    break
            if found_match is not None:
                if det["score"] > merged[found_match]["score"]:
                    merged[found_match] = det
            else:
                merged.append(det)

        # Overlay bounding boxes on the frame
        frame_with_boxes = overlay_bounding_boxes(frame, merged)
        results_per_frame.append(merged)
        
        # Store processed frame
        if is_video:
            processed_frames.append(frame_with_boxes)
        else:
            save_image(frame_with_boxes, "detected_humans.jpg")

    # If it's video and we have processed frames, save them
    if is_video and processed_frames:
        output_path = save_video(processed_frames, "detected_humans.mp4")

    return results_per_frame
