import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/yolov5-detr-FDPN-DASI-118/weights/best.pt') # select your model.pt path
    model.predict(source='dataset/datasets/test/images/',
                  project='runs/detect',
                  name='yolov5-detr-FDPN-DASI-118-test',
                  save=True,
                #   visualize=True # visualize model features maps
                  )
