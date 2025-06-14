import os
from convert_image import process_table_image_to_bw
from detect_line import visualize_detection, get_cropped_images_as_base64

def run_pipeline():

    input_image_path = "/home/hiepnd72/Documents/work/table/image/59b58652bec00a9e53d115.jpg"

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
        base64_list = get_cropped_images_as_base64(
            detection_image_path=processed_bw_image_path,
            original_image_path=input_image_path,
            output_dir=output_rows_dir
        )
        return base64_list
    except Exception as e:
        print(f"Lỗi khi trích xuất ảnh và chuyển đổi base64: {e}")
        return None


if __name__ == "__main__":
    final_base64_list = run_pipeline()
    if final_base64_list:
        for i, b64_string in enumerate(final_base64_list):
            print(f"Ảnh {i+1}: {b64_string[:80]}...")
