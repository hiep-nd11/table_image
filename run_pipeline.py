
import os
from convert_image import process_table_image_to_bw
from detect_line import visualize_detection, extract_rows_intersecting_first_v_line

def run_pipeline():

    input_image_path = "/home/hiepnd72/Documents/work/table/image/59b58652bec00a9e53d115.jpg"

    if not os.path.exists(input_image_path):

        return

    processed_bw_image_path = "table_image_bw_processed.jpg"

    output_rows_dir = "final_cropped_rows"

    try:
        process_table_image_to_bw(input_image_path, processed_bw_image_path)
    except Exception as e:
        return  

    try:
        visualize_detection(processed_bw_image_path)
    except Exception as e:
        return

    try:
        extract_rows_intersecting_first_v_line(
            detection_image_path=processed_bw_image_path,
            original_image_path=input_image_path,
            output_dir=output_rows_dir
        )
    except Exception as e:
        return


if __name__ == "__main__":
    run_pipeline() 