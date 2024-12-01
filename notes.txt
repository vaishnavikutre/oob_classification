# Yolov8 model for inbody-outbody classification


Installation:
create venv (python) - .venv
ultralytics
uninstall opencv-python
install it again
install roboflow
download dataset(create dataset as required...)

/content/data/

train/
asian/
-images/
latino/
-images/
white/
-images/
.....
/content/data/

test/
asian/
-images/
latino/
-images/
white/
-images/

train the model
convert to torchscript
make sure cuda is compatible(pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 )
run the inference

## To run the inference script:
```
python C:\company\Yolov8classification\roboflow\scripts\torchscript_inference.py --model_path C:\company\Yolov8classification\roboflow\runs\classify\train3\weights\best.torchscript --video_path C:/company/Yolov8/classification/oob_videos/OP4-out-8.mp4 --output_path C:/company/Yolov8classification/roboflow/outputs/out.mp4

```

### References:

- https://github.com/roboflow/notebooks/blob/main/notebooks/train-yolov8-classification-on-custom-dataset.ipynb
- https://github.com/taifyang/yolo-inference/blob/main/python/backends/OpenCV/yolo_opencv.py
- https://docs.ultralytics.com/tasks/classify/#how-do-i-validate-a-trained-yolo11-classification-model
- https://www.kaggle.com/datasets/activsurgical/oob-dataset?select=OP4-out-8.mp4
