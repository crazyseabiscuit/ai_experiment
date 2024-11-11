import cv2
import numpy as np
import easyocr

def extract_signature(image_path, output_path):
    # 初始化 EasyOCR 阅读器
    reader = easyocr.Reader(['en'])  # 英文，可以根据需要调整语言

    # 读取图片
    image = cv2.imread(image_path)

    # 使用 EasyOCR 进行文字检测
    result = reader.readtext(image)

    # 寻找签名区域
    signature_box = None
    for (bbox, text, prob) in result:
        if 'signature' in text.lower() or 'signed' in text.lower():
            # 假设签名旁边有标注 "Signature" 或 "Signed"
            signature_box = bbox
            break

    # 如果没有找到明确标注的签名，则尝试通过位置或大小等特征来猜测签名的位置
    if signature_box is None and len(result) > 0:
        # 假设签名位于文档的底部，且宽度较小
        signature_box = sorted(result, key=lambda x: x[0][1], reverse=True)[0][0]

    if signature_box is not None:
        # 裁剪签名区域
        top_left = tuple(signature_box[0])
        bottom_right = tuple(signature_box[2])
        signature = image[int(top_left[1]):int(bottom_right[1]), int(top_left[0]):int(bottom_right[0])]

        # 保存签名图片
        cv2.imwrite(output_path, signature)
        print(f'Signature saved to {output_path}')
    else:
        print('No signature found.')

# 使用函数
image_path = 'path_to_your_image.jpg'
output_path = 'signature.jpg'
extract_signature(image_path, output_path)
