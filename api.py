import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import uvicorn
import argparse
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path  # Thêm thư viện pathlib
from predictor import predictor 
import requests
from PIL import Image
import numpy as np
import pandas as pd
import cv2 as cv2
import io
from pydantic import BaseModel 

class inputImage(BaseModel):
    image: str

class App:
    def __init__(self) -> None:
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.get("/")
        async def root():
            return {"message": "hello"}

        @self.app.post("/image-search")
        async def image_search(image: inputImage):
            image_path = image.image
            print(image_path)

            response = requests.get(image_path)
            
            # Kiểm tra xem tải hình ảnh có thành công không
            if response.status_code == 200:
                # Đọc dữ liệu hình ảnh từ nội dung của response và chuyển nó thành hình ảnh PIL
                image_data = Image.open(io.BytesIO(response.content))
            else:
                print("Không thể tải hình ảnh từ URL.")  
                # Trích xuất đường dẫn tới hình ảnh từ dữ liệu
            store_path = None
            if response.status_code == 200:
                # Đọc dữ liệu hình ảnh từ nội dung của response và chuyển nó thành hình ảnh PIL
                image_data = Image.open(io.BytesIO(response.content))

                # Chuyển đổi hình ảnh PIL thành mảng NumPy
                img = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
                
                # Lưu hình ảnh vào một đường dẫn cụ thể
                store_path = "path_to_save_image"
                if not os.path.exists(store_path):
                    os.makedirs(store_path)
                
                file_name = os.path.split(image_path)[-1]  # Lấy tên tệp từ URL
                dst_path = os.path.join(store_path, file_name)
                
                # Kiểm tra xem mảng hình ảnh có dữ liệu hay không trước khi lưu
                if img is not None and img.size != 0:
                    cv2.imwrite(dst_path, img)
                    # Kiểm tra xem thư mục đã được tạo và hình ảnh đã được lưu không
                    print("Hình ảnh đã được lưu tại:", dst_path)
                else:
                    print("Mảng hình ảnh trống hoặc không hợp lệ.")
            else:
                print("Không thể tải hình ảnh từ URL.")
  
            csv_path = Path('/Users/maixuanduy0605/PycharmProjects/pythonProject1/class_dict.csv')  # Thay đổi đường dẫn tuyệt đối tới class_dict.csv
            model_path = Path('/Users/maixuanduy0605/PycharmProjects/pythonProject1/EfficientNetB3-shoes-87.99.h5')  # Thay đổi đường dẫn tuyệt đối tới model
            klass, prob, img, df  = predictor(store_path, csv_path, model_path, averaged=True, verbose=False)
            print(klass)

            print(f'{prob * 100: 6.2f}')
            # Xóa hết các tệp hình ảnh trong thư mục
            if store_path is not None:
                for filename in os.listdir(store_path):
                    file_path = os.path.join(store_path, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"Không thể xóa tệp {file_path}: {e}")
            return klass

    def run(self, port: int):
        uvicorn.run(self.app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = App()
    app.run(port=args.port)
