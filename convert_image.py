import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_table_image_to_bw(input_image_path, output_image_path):
    """
    Chuyển đổi ảnh bảng màu sang ảnh đen trắng (chữ đen, nền trắng).

    Args:
        input_image_path (str): Đường dẫn đến file ảnh bảng đầu vào.
        output_image_path (str): Đường dẫn để lưu ảnh đen trắng đã xử lý.
    """
    try:
        # 1. Đọc ảnh
        img = cv2.imread(input_image_path)

        if img is None:
            print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {input_image_path}")
            return

        # 2. Chuyển ảnh sang thang độ xám (grayscale)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Áp dụng ngưỡng nhị phân thích ứng (Adaptive Thresholding)
        # Đây là phương pháp hiệu quả cho ảnh tài liệu có độ sáng không đều.
        # Tham số:
        #   - src: Ảnh xám đầu vào.
        #   - maxValue: Giá trị gán cho các pixel lớn hơn ngưỡng (thường là 255 cho màu trắng).
        #   - adaptiveMethod: Phương pháp tính ngưỡng (ADAPTIVE_THRESH_GAUSSIAN_C hoặc ADAPTIVE_THRESH_MEAN_C).
        #                     Gaussian_C tốt hơn vì nó tính trọng số các pixel lân cận theo hàm Gaussian.
        #   - thresholdType: Loại ngưỡng (THRESH_BINARY để tạo ra ảnh đen trắng).
        #   - blockSize: Kích thước của vùng lân cận mà ngưỡng được tính. Phải là số lẻ (ví dụ: 11, 15, 21).
        #                Giá trị càng lớn thì ngưỡng càng "mịn", giá trị nhỏ hơn thì chi tiết hơn.
        #   - C: Hằng số trừ đi từ giá trị trung bình/trọng số Gaussian. Có thể là số âm hoặc dương.
        #        Dùng để điều chỉnh độ sáng chung sau khi áp dụng ngưỡng.
        
        # Thử nghiệm với các giá trị blockSize và C để có kết quả tốt nhất.
        # Ví dụ: blockSize=11, C=2
        # Một số giá trị khác bạn có thể thử: blockSize=21, C=10 hoặc blockSize=31, C=5
        
        # Thử nghiệm với các tham số khác nhau
        thresh_img_adaptive = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
        )

        # *************** HOẶC ***************
        # 3b. Áp dụng ngưỡng Otsu's Binarization
        # Phương pháp này tự động tìm một ngưỡng toàn cục tối ưu nếu histogram của ảnh có 2 đỉnh rõ rệt.
        # Nó phù hợp cho ảnh có độ sáng tương đối đồng đều.
        # blur = cv2.GaussianBlur(gray_img, (5,5), 0) # Làm mờ nhẹ để giảm nhiễu trước khi áp dụng Otsu
        # ret, thresh_img_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # print(f"Ngưỡng Otsu tự động tìm được: {ret}")
# image_to_save = thresh_img_otsu
        # ************************************

        image_to_save = thresh_img_adaptive

        # 4. Lưu ảnh đen trắng
        cv2.imwrite(output_image_path, image_to_save)

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
