import cv2
import numpy as np

from itertools import combinations
import math


# Euclidean Distance between two points
def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# Load Yolo
yolo_weight = "data/model/yolov3.weights"
yolo_config = "data/model/yolov3.cfg"
coco_labels = "data/model/coco.names"
net = cv2.dnn.readNet(yolo_weight, yolo_config)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


classes = []
with open(coco_labels, "r") as f:
    classes = [line.strip() for line in f.readlines()]


layer_names = list(net.getLayerNames())
output_layers = [layer_names[i - 1] for i in list(net.getUnconnectedOutLayers())]


# Below function will read video frames
cap = cv2.VideoCapture('data/cctv.mp4')


while True:

    read_ok, img = cap.read()

    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = [0]  # 0 is for person detection
    confidences = []
    boxes = []
    center_points = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # class_id = 0 means we will only detect persons from video
            if confidence > 0.5 and class_id == 0:
                
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h, center_x, center_y])
                center_points.append([center_x, center_y])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    
    # Create combination list of center points between each detected person bounding box
    combination_points = list(combinations(center_points, 2))
    
    font = cv2.FONT_HERSHEY_DUPLEX
    
    for i in range(len(boxes)):
        
        #if i in indexes:
        x, y, w, h, box_center_x, box_center_y = boxes[i]

        
        #cv2.circle(img, (center_x, center_y), 3, (0, 0, 255), cv2.FILLED)

        for points in combination_points:
            
            # Find Distance between two person (pixel distance / apart)
            center_x, center_y = points[0]
            prev_center_x, prev_center_y = points[1]
            euclidean_distance = calculateDistance(center_x, center_y, prev_center_x, prev_center_y)
            
            # Width of three tiles = 217 pixel , which is 9 feet
            width_of_3_tiles = 300 #217

            # Mark person bounding box as red is distance is less than 9 feet
            if euclidean_distance < width_of_3_tiles and euclidean_distance > 100:
                if box_center_x == center_x or box_center_y == center_y:
                    # Draw rectangle for each person
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # Draw line between each person
                    cv2.line(img, (center_x, center_y), (prev_center_x, prev_center_y), (0, 255, 0), thickness=2)
                    #cv2.putText(img, str(int(euclidean_distance)), (int((center_x+prev_center_x)/2), int((center_y + prev_center_y)/2)), font, 1, (255, 0, 0), thickness=2)
                    
    
    cv2.imshow("Image", img)
    # Close video window by pressing 'x'
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

