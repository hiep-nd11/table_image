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

def extract_cells(image_path, output_dir="cells"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    img, h_lines, v_lines = detect_table_lines(image_path)

    h_lines.insert(0, (0, 0, img.shape[1], 1))
    h_lines.append((0, img.shape[0]-1, img.shape[1], 1))

    v_lines.insert(0, (0, 0, 1, img.shape[0]))
    v_lines.append((img.shape[1]-1, 0, 1, img.shape[0]))

    y_coords = [line[1] for line in h_lines]
    x_coords = [line[0] for line in v_lines]

    cells = []
    rows = []

    for i in range(len(y_coords) - 1):
        row_cells = []
        y1, y2 = y_coords[i], y_coords[i+1]

        if y2 - y1 > 20:  
            for j in range(len(x_coords) - 1):
                x1, x2 = x_coords[j], x_coords[j+1]

                if x2 - x1 > 30:  
                    cell_img = img[y1:y2, x1:x2]
                    cells.append(cell_img)
                    row_cells.append(cell_img)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cv2.imwrite(f"{output_dir}/{timestamp}_cell_r{i:02d}_c{j:02d}.png",
                               cv2.cvtColor(cell_img, cv2.COLOR_RGB2BGR))

            if row_cells:
                rows.append(row_cells)
                row_img = img[y1:y2, :]
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(f"{output_dir}/{timestamp}_row_{i:02d}.png",
                           cv2.cvtColor(row_img, cv2.COLOR_RGB2BGR))

    return cells, rows

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

def extract_first_column_cells(image_path, output_dir="first_column_cells"):

    import os
    os.makedirs(output_dir, exist_ok=True)

    img_rgb, h_lines, v_lines, method = auto_detect_table(image_path)

    if img_rgb is None:
        return 

    if len(v_lines) < 2:
        return
    if len(h_lines) < 2:
        return

    h_lines.sort(key=lambda line: line[1]) 
    v_lines.sort(key=lambda line: line[0]) 

    x1 = v_lines[0][0]
    x2 = v_lines[1][0]

    cropped_count = 0
    for i in range(len(h_lines) - 1):
        y1 = h_lines[i][1] + h_lines[i][3] 
        y2 = h_lines[i+1][1] 

        if y2 - y1 < 10: 
            continue

        cell_img = img_rgb[y1:y2, x1:x2]

        if cell_img.size == 0:
            continue

        cell_img_bgr = cv2.cvtColor(cell_img, cv2.COLOR_RGB2BGR)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = f"{output_dir}/{timestamp}_first_col_cell_row_{i+1:02d}.png"
        cv2.imwrite(output_path, cell_img_bgr)
        cropped_count += 1

    if cropped_count > 0:
        return
    else:
        return

def extract_intersecting_rows(image_path, output_dir="intersecting_rows"):

    import os
    os.makedirs(output_dir, exist_ok=True)

    img_rgb, h_lines, v_lines, method = auto_detect_table(image_path)
    if img_rgb is None:
        return

    if len(v_lines) < 2 or len(h_lines) < 2:
        return

    v_lines.sort(key=lambda line: line[0])
    h_lines.sort(key=lambda line: line[1])

    first_col_x_start = v_lines[0][0]
    first_col_x_end = v_lines[1][0]

    intersecting_h_lines = []
    for x, y, w, h in h_lines:
        line_x_start = x
        line_x_end = x + w
        if line_x_start < first_col_x_end and first_col_x_start < line_x_end:
            intersecting_h_lines.append((x, y, w, h))

    if len(intersecting_h_lines) < 2:
        return

    img_height, img_width, _ = img_rgb.shape
    cropped_count = 0
    for i in range(len(intersecting_h_lines) - 1):
        y1 = intersecting_h_lines[i][1]
        y2 = intersecting_h_lines[i+1][1]

        if y2 - y1 < 5:
            continue

        row_img = img_rgb[y1:y2, 0:img_width]

        if row_img.size > 0:
            row_img_bgr = cv2.cvtColor(row_img, cv2.COLOR_RGB2BGR)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = f"{output_dir}/{timestamp}_row_{i+1:02d}.png"
            cv2.imwrite(output_path, row_img_bgr)
            cropped_count += 1

    if cropped_count > 0:
        return
    else:
        return

def extract_rows_intersecting_first_v_line(detection_image_path, original_image_path, output_dir="final_cropped_rows"):

    get_cropped_images_as_base64(detection_image_path, original_image_path, output_dir=output_dir)

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

    slice_images = []
    if len(sorted_cut_points) > 1:
        for i in range(len(sorted_cut_points) - 1):
            slice_y1 = sorted_cut_points[i]
            slice_y2 = sorted_cut_points[i+1]

            if slice_y2 - slice_y1 < 5:
                continue

            slice_img = original_img_rgb[slice_y1:slice_y2, 0:img_width]
            
            if slice_img.size > 0:
                slice_img_bgr = cv2.cvtColor(slice_img, cv2.COLOR_RGB2BGR)
                slice_images.append(slice_img_bgr)
    
    vertical_cut_image_bgr = None
    if len(v_lines) >= 2:
        target_vertical_line_for_cut = v_lines[-2]
        cut_x = target_vertical_line_for_cut[0]

        vertical_cut_image = original_img_rgb[:, cut_x:]

        if vertical_cut_image.size > 0:
            vertical_cut_image_bgr = cv2.cvtColor(vertical_cut_image, cv2.COLOR_RGB2BGR)
            

    else:
        print(f"Không đủ đường kẻ dọc (cần ít nhất 2, tìm thấy {len(v_lines)}) để thực hiện cắt dọc bổ sung.")

    return {"slices": slice_images, "vertical_cut": vertical_cut_image_bgr}


def get_cropped_images_as_base64(detection_image_path, original_image_path, output_dir=None):

    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)

    cropped_images = _get_cropped_images(detection_image_path, original_image_path)
    base64_list = []

    slice_images = cropped_images["slices"]
    vertical_cut_image = cropped_images["vertical_cut"]

    for i, img in enumerate(slice_images):
        if output_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{output_dir}/{timestamp}_slice_{i+1:02d}.png"
            cv2.imwrite(output_path, img)

        _, buffer = cv2.imencode('.png', img)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        base64_list.append(base64_str)

    if vertical_cut_image is not None:
        if output_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path_vertical = f"{output_dir}/{timestamp}_vertical.png"
            cv2.imwrite(output_path_vertical, vertical_cut_image)

        _, buffer = cv2.imencode('.png', vertical_cut_image)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        base64_list.append(base64_str)

    return base64_list