from yolov5_tflite_inference import yolov5_tflite
import argparse
import cv2
from PIL import Image
from utils import letterbox_image, scale_coords
import numpy as np
import time
import matplotlib.pyplot as plt
from sort import *


tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [5, 195, 281, 195]

totalCount = []

def detect_video(weights,img_size,conf_thres,iou_thres):

    start_time = time.time()

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoCapture('rtsp://admin:ga0512117$@192.168.23.11:554/cam/realmonitor?channel=1&subtype=1')
    video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    fps = video.get(cv2.CAP_PROP_FPS)
    h = int(video.get(3))
    w = int(video.get(4))
    print(w,h)
    #h = 1280
    #w = 720
    #result_video_filepath = video_filepath.split('/')[-1].split('.')[0] + 'yolov5_output.mp4'
    #out  = cv2.VideoWriter(result_video_filepath,fourcc,int(fps),(288, 288), 0)

    yolov5_tflite_obj = yolov5_tflite(weights,img_size,conf_thres,iou_thres)

    size = (img_size,img_size)
    no_of_frames = 0
    
    while True:
        video.grab()
    
        check, frame = video.read()
        
        if not check:
            break
        frame = cv2.resize(frame, size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        current_buffer_size = video.get(cv2.CAP_PROP_BUFFERSIZE)
        print(f"Tamanho atual do buffer de frames: {current_buffer_size}")


        
        no_of_frames += 1
        image_resized = letterbox_image(Image.fromarray(frame), size)
        image_array = np.asarray(image_resized)

        normalized_image_array = image_array.astype(np.float32) / 255.0
        result_boxes, result_scores, result_class_names = yolov5_tflite_obj.detect(normalized_image_array)
        
        detections = np.empty((0, 5))
        
        if len(result_boxes) > 0:
          
          result_boxes = scale_coords(size,np.array(result_boxes), size)
          font = cv2.FONT_HERSHEY_SIMPLEX 
          
          # org 
          #org = (10, 20) 
              
          # fontScale 
          fontScale = 0.3
              
          # Blue color in BGR 
          color = (0, 255, 0) 
              
          # Line thickness of 1 px 
          thickness = 1
  
          for i,r in enumerate(result_boxes):
          
              x1, y1 = int(r[0]), int(r[1])
              x2, y2 = int(r[2]), int(r[3])
              
              currentArray = np.array([x1, y1, x2, y2, int(100*result_scores[i])])
              detections = np.vstack((detections, currentArray))
              
              
              org = (int(r[0]),int(r[1]))
              cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
              cv2.putText(frame, str(int(100*result_scores[i])) + '%  ' + str(result_class_names[i]), org, font,  
                          fontScale, color, thickness, cv2.LINE_AA)
                          
              
          
          resultsTracker = tracker.update(detections)
  
          cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 2)
  
          
          for result in resultsTracker:
              x1, y1, x2, y2, id = result
      
              wb, hb = int(x2) - int(x1), int(y2) - int(y1)
              
              cx, cy = int(x1 + wb // 2), int(y1 + hb // 2)  # Converta para int
              
              cv2.circle(frame, (cx, cy), 2, (255, 0, 255), cv2.FILLED)
              
  
              if limits[0] < cx < limits[2] and limits[1] - 5 < cy < limits[1] + 5:
                  if totalCount.count(id) == 0:
                      totalCount.append(id)
                      cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)
                      cv2.circle(frame, (cx, cy), 2, (0, 255, 0), cv2.FILLED)
      
          # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
          cv2.putText(frame, str(len(totalCount)),(50, 50),cv2.FONT_HERSHEY_PLAIN,1,(50,50,255),2)
        
        
        # Exibir o frame em uma janela
        cv2.imshow('YOLOv5 Real-Time Detection', frame)
        
        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        #out.write(frame)
        end_time = time.time()
        print('FPS:',1/(end_time-start_time))
        start_time = end_time
    #out.release()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--weights', type=str, default='288-2.tflite', help='model.tflite path(s)')
    #parser.add_argument('-v','--video_path', type=str, required=True, help='video path')  
    parser.add_argument('--img_size', type=int, default=288, help='image size') 
    parser.add_argument('--conf_thres', type=float, default=0.9, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.9, help='IOU threshold for NMS')

    
    opt = parser.parse_args()
    
    print(opt)
    detect_video(opt.weights,opt.img_size,opt.conf_thres,opt.iou_thres)