import os
from convert_image import process_table_image_to_bw
from detect_line import visualize_detection, get_processed_image_data

def run_pipeline():

    input_image_path = "/home/hiepnd72/Documents/work/table/image/51099de01dbcaae2f3ad.jpg"

    if not os.path.exists(input_image_path):
        print(f"Lỗi: không tìm thấy file ảnh đầu vào tại: {input_image_path}")
        return None

    processed_bw_image_path = "table_image_bw_processed.jpg"

    output_rows_dir = "final_cropped_rows"

    try:
        process_table_image_to_bw(input_image_path, processed_bw_image_path)
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh sang đen trắng: {e}")
        return None

    try:
        visualize_detection(processed_bw_image_path)
    except Exception as e:
        print(f"Lỗi khi phát hiện đường kẻ: {e}")
        return None

    try:
        processed_data = get_processed_image_data(
            detection_image_path=processed_bw_image_path,
            original_image_path=input_image_path,
            output_dir=output_rows_dir
        )
        return processed_data
    except Exception as e:
        print(f"Lỗi khi trích xuất ảnh và chuyển đổi base64: {e}")
        return None


if __name__ == "__main__":
    final_processed_data = run_pipeline()
    if final_processed_data:
        
        h_slices = final_processed_data.get("horizontal_slices", [])
        for i, slice_details in enumerate(h_slices):
            merged_count = slice_details.get("merged_rows", 1)
            b64_preview = slice_details['image_b64'][:60]

        v_slice = final_processed_data.get("vertical_slice")

