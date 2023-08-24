# multi-model_app_2
This is a test of multi-model_app.

updated version of multi-model_app_1:
https://github.com/lifang535/multi-model_app_1/tree/main

## 更新内容

`Model_4` 和 `Model_5` 合并，并用 `request.sub_requests` 保存的信息代替特殊信号（单个帧或单个视频处理完成）

---

`Model_2` 使用 `easyocr` 库进行文本识别（车牌号检测）：

https://github.com/JaidedAI/EasyOCR

单独使用模型时可以起到对清晰照片的车牌号检测效果，但是暂时没有找到合适的视频

并且使用 `easyocr` 库更容易造成 `cuda out of memory`，减少进程但运行时 `Model_2` 排队比较严重

---

`Model_3` 使用 `cv2.face.EigenFaceRecognizer_create()` 进行人脸识别，识别对象为 `'face_data'` 里的内容，但模型未调用 GPU，速度快但效果较差

## TODO

1. `Model_3` 人脸识别模型效果较差

2. 代码内存占用问题
