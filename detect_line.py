import cv2
import numpy as np
from matplotlib import pyplot as plt
import datetime
import base64

def detect_table_lines(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    h_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    v_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_width = img.shape[1] * 0.3
    valid_h_lines = []

    for contour in h_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > min_width and h < 10:  
            valid_h_lines.append((x, y, w, h))

    min_height = img.shape[0] * 0.2
    valid_v_lines = []

    for contour in v_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > min_height and w < 10:  
            valid_v_lines.append((x, y, w, h))

    valid_h_lines.sort(key=lambda x: x[1])  
    valid_v_lines.sort(key=lambda x: x[0])  

    return img_rgb, valid_h_lines, valid_v_lines


def visualize_detection(image_path):
    img, h_lines, v_lines = detect_table_lines(image_path)

    img_with_lines = img.copy()

    for x, y, w, h in h_lines:
        cv2.rectangle(img_with_lines, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img_with_lines, f"H:y={y}", (x+10, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    for x, y, w, h in v_lines:
        cv2.rectangle(img_with_lines, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_with_lines, f"V:x={x}", (x+5, y+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return h_lines, v_lines


def advanced_table_detection(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY_INV, 15, 10)

    horizontal_kernel = np.ones((1, 50), np.uint8)
    horizontal_lines = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, horizontal_kernel)

    vertical_kernel = np.ones((50, 1), np.uint8)
    vertical_lines = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, vertical_kernel)

    horizontal_projection = np.sum(horizontal_lines, axis=1)
    h_line_positions = []

    h_threshold = np.max(horizontal_projection) * 0.1
    for i, val in enumerate(horizontal_projection):
        if val > h_threshold:
            h_line_positions.append(i)

    vertical_projection = np.sum(vertical_lines, axis=0)
    v_line_positions = []

    v_threshold = np.max(vertical_projection) * 0.1
    for i, val in enumerate(vertical_projection):
        if val > v_threshold:
            v_line_positions.append(i)

    grouped_h_lines = []
    if h_line_positions:
        current_group = [h_line_positions[0]]
        for pos in h_line_positions[1:]:
            if pos - current_group[-1] < 10:
                current_group.append(pos)
            else:
                grouped_h_lines.append(int(np.mean(current_group)))
                current_group = [pos]
        grouped_h_lines.append(int(np.mean(current_group)))

    grouped_v_lines = []
    if v_line_positions:
        current_group = [v_line_positions[0]]
        for pos in v_line_positions[1:]:
            if pos - current_group[-1] < 10:
                current_group.append(pos)
            else:
                grouped_v_lines.append(int(np.mean(current_group)))
                current_group = [pos]
        grouped_v_lines.append(int(np.mean(current_group)))

    return img, grouped_h_lines, grouped_v_lines

def auto_detect_table(image_path):
    try:
        img, h_lines, v_lines = detect_table_lines(image_path)
        if len(h_lines) >= 2 and len(v_lines) >= 2:
            return img, h_lines, v_lines, "basic"
    except:
        pass

    try:
        img, h_lines, v_lines = advanced_table_detection(image_path)

        h_lines_formatted = [(0, y, img.shape[1], 1) for y in h_lines]
        v_lines_formatted = [(x, 0, 1, img.shape[0]) for x in v_lines]

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), h_lines_formatted, v_lines_formatted, "advanced"
    except:
        return None, [], [], "failed"

def extract_rows_intersecting_first_v_line(detection_image_path, original_image_path, output_dir="final_cropped_rows"):

    get_processed_image_data(detection_image_path, original_image_path, output_dir=output_dir)

def _get_cropped_images(detection_image_path, original_image_path):
    _, all_h_lines, v_lines, method = auto_detect_table(detection_image_path)

    if not v_lines:
        print("Không tìm thấy đường kẻ dọc.")
        return {"slices": [], "vertical_cut": None}
    if len(all_h_lines) < 2:
        return {"slices": [], "vertical_cut": None}

    original_img_bgr = cv2.imread(original_image_path)
    if original_img_bgr is None:
        print("Không thể đọc ảnh gốc.")
        return {"slices": [], "vertical_cut": None}
    original_img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)

    v_lines.sort(key=lambda line: line[0])

    if len(v_lines) < 2:
        print("Không đủ đường kẻ dọc để xác định cột đầu tiên.")
        return {"slices": [], "vertical_cut": None}

    border_line = v_lines[0]

    target_v_line = None
    for candidate_line in v_lines[1:]:
        distance = candidate_line[0] - border_line[0]
        if distance >= 10:
            target_v_line = candidate_line
            break  

    if target_v_line is None:
        print("Không tìm thấy đường kẻ dọc phù hợp.")
        return {"slices": [], "vertical_cut": None}

    first_v_line_x = target_v_line[0]

    intersecting_h_lines = []
    for x, y, w, h in all_h_lines:
        line_x_start = x - 10
        line_x_end = x + w + 10
        if line_x_start <= first_v_line_x and first_v_line_x <= line_x_end:
            intersecting_h_lines.append((x, y, w, h))

    intersecting_h_lines.sort(key=lambda line: line[1])

    special_row_boundaries = []
    if len(intersecting_h_lines) >= 2:
        for i in range(len(intersecting_h_lines) - 1):
            y1 = intersecting_h_lines[i][1]
            y2 = intersecting_h_lines[i+1][1]

            lines_in_between = 0
            for line in all_h_lines:
                if y1 < line[1] < y2:
                    lines_in_between += 1
            
            if lines_in_between > 1:
                special_row_boundaries.append((y1, y2))

    img_height, img_width, _ = original_img_rgb.shape
    final_cut_points = {0, img_height} 
    for start, end in special_row_boundaries:
        final_cut_points.add(start)
        final_cut_points.add(end)
    
    sorted_cut_points = sorted(list(final_cut_points))

    slice_details = []
    if len(sorted_cut_points) > 1:
        for i in range(len(sorted_cut_points) - 1):
            slice_y1 = sorted_cut_points[i]
            slice_y2 = sorted_cut_points[i+1]

            if slice_y2 - slice_y1 < 5:
                continue

            lines_inside = 0
            for line in all_h_lines:
                if slice_y1 < line[1] < slice_y2:
                    lines_inside += 1
            
            is_single_row = (lines_inside == 0)

            slice_img = original_img_rgb[slice_y1:slice_y2, 0:img_width]
            
            if slice_img.size > 0:
                slice_details.append({
                    "is_single_row": is_single_row,
                    "y_start": slice_y1,
                    "y_end": slice_y2
                })

    grouped_slices = []
    i = 0
    while i < len(slice_details):
        current_slice_info = slice_details[i]
        if not current_slice_info['is_single_row']:
            grouped_slices.append({
                "type": "complex",
                "slices": [current_slice_info],
                "y_start": current_slice_info['y_start'],
                "y_end": current_slice_info['y_end']
            })
            i += 1
        else:
            group_start_index = i
            j = i
            while j < len(slice_details) and slice_details[j]['is_single_row']:
                j += 1
            
            group_end_index = j - 1
            group_slices = slice_details[group_start_index:j]
            
            grouped_slices.append({
                "type": "single_group",
                "slices": group_slices,
                "y_start": group_slices[0]['y_start'],
                "y_end": group_slices[-1]['y_end']
            })
            
            i = j
    
    final_slices = []
    first_merged_group = None
    isolated_singles = []
    
    for group in grouped_slices:
        if group["type"] == "single_group" and len(group["slices"]) > 1:
            if first_merged_group is None:
                first_merged_group = group
            else:
                final_slices.append(group)
        elif group["type"] == "single_group" and len(group["slices"]) == 1:
            isolated_singles.extend(group["slices"])
        else:
            final_slices.append(group)
    
    if isolated_singles:
        if first_merged_group is not None:
            original_count = len(first_merged_group["slices"])
            first_merged_group["slices"].extend(isolated_singles)
            first_merged_group["y_end"] = isolated_singles[-1]["y_end"]
        else:
            first_complex_group = None
            for i, group in enumerate(final_slices):
                if group["type"] == "complex":
                    first_complex_group = group
                    final_slices.pop(i)  
                    break
            
            if first_complex_group is not None:
                first_merged_group = {
                    "type": "single_group",
                    "slices": first_complex_group["slices"] + isolated_singles,
                    "y_start": first_complex_group["y_start"],
                    "y_end": isolated_singles[-1]["y_end"]
                }
            else:
                first_merged_group = {
                    "type": "single_group",
                    "slices": isolated_singles,
                    "y_start": isolated_singles[0]["y_start"],
                    "y_end": isolated_singles[-1]["y_end"]
                }
    
    if first_merged_group is not None:
        final_slices.insert(0, first_merged_group)
    
    final_processed_slices = []
    for group in final_slices:
        if group["type"] == "complex":
            slice_info = group["slices"][0]
            y_start = slice_info['y_start']
            y_end = slice_info['y_end']
            img = original_img_rgb[y_start:y_end, 0:img_width]
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            final_processed_slices.append({
                "image": img_bgr,
                "is_single_row": False,
                "y_start": y_start,
                "y_end": y_end,
                "merged_rows": 1
            })
        else:  

            individual_images = []
            overall_y_start = group["slices"][0]["y_start"]
            overall_y_end = group["slices"][-1]["y_end"]
            
            for slice_info in group["slices"]:
                slice_y_start = slice_info["y_start"]
                slice_y_end = slice_info["y_end"]
                slice_img = original_img_rgb[slice_y_start:slice_y_end, 0:img_width]
                individual_images.append(slice_img)
            
            if individual_images:
                merged_img = np.vstack(individual_images)
                merged_img_bgr = cv2.cvtColor(merged_img, cv2.COLOR_RGB2BGR)
                
                num_merged = len(group["slices"])
                final_processed_slices.append({
                    "image": merged_img_bgr,
                    "is_single_row": (num_merged == 1),
                    "y_start": overall_y_start,
                    "y_end": overall_y_end,
                    "merged_rows": num_merged
                })
    
    vertical_cut_image_bgr = None
    if len(v_lines) >= 2:
        target_vertical_line_for_cut = v_lines[-2]
        cut_x = target_vertical_line_for_cut[0]

        vertical_cut_image = original_img_rgb[:, cut_x:]

        if vertical_cut_image.size > 0:
            vertical_cut_image_bgr = cv2.cvtColor(vertical_cut_image, cv2.COLOR_RGB2BGR)
            
    else:
        print(f"Không đủ đường kẻ dọc (cần ít nhất 2, tìm thấy {len(v_lines)}) để thực hiện cắt dọc bổ sung.")

    return {"slices": final_processed_slices, "vertical_cut": vertical_cut_image_bgr}


def get_processed_image_data(detection_image_path, original_image_path, output_dir=None):

    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)

    cropped_data = _get_cropped_images(detection_image_path, original_image_path)
    
    processed_slices = []
    for i, slice_details in enumerate(cropped_data["slices"]):
        img = slice_details["image"]
        
        if output_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{output_dir}/{timestamp}_slice_{i+1:02d}.png"
            cv2.imwrite(output_path, img)

        _, buffer = cv2.imencode('.png', img)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        
        new_slice_info = slice_details.copy()
        new_slice_info.pop("image")
        new_slice_info["image_b64"] = base64_str
        processed_slices.append(new_slice_info)

    processed_vertical_cut = None
    vertical_cut_image = cropped_data["vertical_cut"]
    if vertical_cut_image is not None:
        if output_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path_vertical = f"{output_dir}/{timestamp}_vertical.png"
            cv2.imwrite(output_path_vertical, vertical_cut_image)

        _, buffer = cv2.imencode('.png', vertical_cut_image)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        processed_vertical_cut = {"image_b64": base64_str}

    return {
        "horizontal_slices": processed_slices,
        "vertical_slice": processed_vertical_cut
    }