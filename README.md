# multi-model_app_2
This is a test of multi-model_app.

updated version of multi-model_app_1:
https://github.com/lifang535/multi-model_app_1/tree/main

## 更新内容

`Model_4` 和 `Model_5` 合并，并用 `request.sub_requests` 保存的信息代替特殊信号（单个帧或单个视频处理完成）

---

`Model_2` 使用 `huggingface` 上的几个模型不太符合需求

使用 `from paddleocr import PaddleOCR`：
https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/quickstart.md
模型测试效果较好但是执行完
```
python -m pip install paddlepaddle-gpu==2.5.1.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```
指令后 `pipeline.py` 无法运行，包冲突

最终使用 `easyocr` 库进行文本识别（车牌号检测），单独使用模型时可以起到对清晰照片的车牌号检测效果，但是暂时没有找到合适的视频

并且使用 `easyocr` 库更容易造成 `cuda out of memory`，减少进程但运行时 `Model_2` 排队比较严重

---

debug：

1. 视频帧转数组和转成图像用了不同的库，导致信息丢失，但是更正后视频信息量较大队列堆积

2. 有几辆车几个人就传几次视频帧，造成没有车和人的视频帧没有被传输，改为经过 `Model_3` 向后传

## TODO

1. `Model_3` 人脸识别模型

2. 代码内存占用问题
