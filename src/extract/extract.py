import cv2
import numpy as np
import os
from config import EXTRACT_DIR

EDGE_THRESHOLD = 125
EXPAND_FRAC = 0.2
DEBUG = True


def _to_gray(img):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _edge_map(gray):
    sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edges = (np.abs(sobel) > EDGE_THRESHOLD).astype(np.uint8) * 255
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    if DEBUG:
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        cv2.imwrite(f"{EXTRACT_DIR}/edges.png", edges)
    return edges


def _connect_text_horizontally(edge_map, kernel_width=30, kernel_height=1):
    kernel = np.ones((kernel_height, kernel_width), np.uint8)
    connected = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel, iterations=2)
    return connected


# def _connect_text_vertically(edge_map, kernel_height=15, kernel_width=5):
#     kernel = np.ones((kernel_height, kernel_width), np.uint8)
#     connected = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel, iterations=1)
#     return connected


def _get_bounding_boxes(binary_map, min_area=500, max_aspect_ratio=20):
    binary_map_uint8 = binary_map.astype(np.uint8)
    contours, _ = cv2.findContours(binary_map_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
    
    return sorted(boxes, key=lambda b: (b[1], b[0]))


# def _merge_overlapping_boxes(boxes, x_threshold=20, y_threshold=10):
#     if not boxes:
#         return []
    
#     merged = []
#     boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
#     current = list(boxes[0])
    
#     for box in boxes[1:]:
#         x1, y1, x2, y2 = current
#         nx1, ny1, nx2, ny2 = box
        
#         y_overlap = not (y2 + y_threshold < ny1 or ny2 + y_threshold < y1)
#         x_overlap = not (x2 + x_threshold < nx1 or nx2 + x_threshold < x1)
        
#         if y_overlap and x_overlap:
#             current = [
#                 min(x1, nx1),
#                 min(y1, ny1),
#                 max(x2, nx2),
#                 max(y2, ny2)
#             ]
#         else:
#             merged.append(tuple(current))
#             current = list(box)
    
#     merged.append(tuple(current))
#     return merged


def _add_padding(boxes, img_width, img_height):
    padded = []
    for bx1, by1, bx2, by2 in boxes:
        pad_x = max(1, int((bx2 - bx1) * EXPAND_FRAC // 2))
        pad_y = max(1, int((by2 - by1) * EXPAND_FRAC))
        x1 = max(0, bx1 - pad_x)
        y1 = max(0, by1 - pad_y)
        x2 = min(img_width, bx2 + pad_x)
        y2 = min(img_height, by2 + pad_y)
        padded.append((x1, y1, x2, y2))
    return padded


def extract(image):
    gray = _to_gray(image)
    img_height, img_width = gray.shape
    
    edges = _edge_map(gray)
    
    connected_h = _connect_text_horizontally(edges)
    
    if DEBUG:
        cv2.imwrite(f"{EXTRACT_DIR}/connected_horizontal.png", connected_h)
    
    # connected_full = _connect_text_vertically(connected_h)
    
    # if DEBUG:
    #     cv2.imwrite(f"{EXTRACT_DIR}/connected_full.png", connected_full)
    
    boxes = _get_bounding_boxes(connected_h)
    
    if DEBUG:
        debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(f"{EXTRACT_DIR}/initial_boxes.png", debug_img)
    
    # boxes = _merge_overlapping_boxes(boxes, x_threshold=30, y_threshold=15)
    
    if DEBUG:
        debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(f"{EXTRACT_DIR}/merged_boxes.png", debug_img)
    
    boxes = _add_padding(boxes, img_width, img_height)
    
    if DEBUG:
        debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(f"{EXTRACT_DIR}/final_lines.png", debug_img)
    
    frames = []
    
    for x1, y1, x2, y2 in boxes:
        
        line_crop = image[y1:y2, x1:x2].copy()
        
        if line_crop.size > 0:
            frames.append(line_crop)
    
    print(f"[1] Extraction done: {len(frames)} lines")
    return frames