import os
import numpy as np
from vision_agent.tools import *
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool

def detect_and_count_person(media_path: str) -> str:
    """
    Detects and counts persons from a media (image or video) using the tools described:
      - For images: 'countgd_object_detection'
      - For videos: 'florence2_sam2_video_tracking'
    Overlays bounding boxes or segmentation masks on the detections and returns the path
    to the saved annotated file.

    Parameters:
        media_path (str): The path or URL to an image or video.

    Returns:
        str: The file path for the annotated image or video.
    """

    import os

    # Try loading as image first
    try:
        image = load_image(media_path)  # load_image from the documentation
        # If successful, run single-image detection
        detections = countgd_object_detection("person", image)

        # Overlay bounding boxes
        annotated_image = overlay_bounding_boxes(image, detections)
        output_path = "annotated_image_result.png"
        save_image(annotated_image, output_path)
        return output_path

    except Exception:
        # If loading as image failed, assume it's a video
        # Extract frames at default fps
        frames_and_timestamps = extract_frames_and_timestamps(media_path)
        frames = [ft["frame"] for ft in frames_and_timestamps]

        # Track persons across frames
        tracked_objects = florence2_sam2_video_tracking("person", frames)

        # Overlay bounding boxes from the tracking data
        annotated_frames = []
        for frame, frame_dets in zip(frames, tracked_objects):
            annotated_frame = overlay_bounding_boxes(frame, frame_dets)
            annotated_frames.append(annotated_frame)

        # Save the annotated frames as a video
        output_video_path = "annotated_video_result.mp4"
        final_video_path = save_video(annotated_frames, output_video_path)
        return final_video_path
