import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('/home/s317/users/lfq/RTDETR-main/ultralytics/cfg/models/yolo-detr/yolov5-detr-FDPN-DASI.yaml')
    model.load('/home/s317/users/lfq/RTDETR-main/weights/yolov5su.pt') # loading pretrain weights
    model.train(data='/home/s317/users/lfq/RTDETR-main/dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=4,
                workers=4,
                device='0',
                # resume='', # last.pt path
                project='runs/train',
                name='yolov5-detr-FDPN-DASI-118',
                )
