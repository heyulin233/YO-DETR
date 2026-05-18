import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/yolov5-detr-FDPN-DASI-118/weights/best.pt')
    model.val(data='dataset/data.yaml',
              split='val',
              imgsz=640,
              batch=4,
            #   save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='yolov5-detr-FDPN-DASI-118-train',
              )
