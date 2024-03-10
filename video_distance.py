from imutils import paths
import numpy as np
import imutils
import cv2
import os
import ultralytics
from ultralytics import YOLO
import time
import torch
import utils
import ssl

def bounding_box(img, model):
    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Run prediction using model on image
    results = model(image_rgb)
    # Get the bounding boxes and confidence scores
    boxes = results[0].boxes.xyxy.cpu().numpy()
    conf = results[0].boxes.conf.tolist()
    # Create copy of boxes
    image_with_boxes = img.copy()
    # Map the bounding boxes onto the image
    return boxes, conf

def distance_to_camera(focal_length_mm,real_object_height_mm,image_height_px,object_height_px,sensor_height_mm):
      d = (focal_length_mm * real_object_height_mm * image_height_px)/(object_height_px*sensor_height_mm) 
      return d

def find_distances(img,model):
    bboxes, conf = bounding_box(img,model)
    cones = []
    for i in range(len(bboxes)):
        cone = []
        cone.append([int(i) for i in bboxes[i]])
        object_height_px = bboxes[i][3] - bboxes[i][1]
        focalLength = 24
        cone_height = 12*25.4 # 12 inches * 25.4 to convert to mm
        sensor_height = 14.4
        image_height = img.shape[0]
        distance = distance_to_camera(focalLength,cone_height,image_height,object_height_px,sensor_height)
        cone.append(distance)
        cone.append(conf[i])
        cones.append(cone)
    return cones
    
def display(cones,img,show):
    for cone in cones:
        # does not display if confidence is below conf_thresh
        conf_thresh = 0.4
        if cone[2] < conf_thresh:
            continue
        x1, y1, x2, y2 = cone[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Draw the bounding box on the image
        color = (0, 255, 0) 
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        note = "{:.2f}".format(cone[1]/25.4) + "in. " + "conf:" + "{:.2f}".format(cone[2])
        # writes the distance and confidence
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1 
        font_thickness = 4
        text_size = cv2.getTextSize(note, font, font_scale, font_thickness)[0]
        text_position = (cone[0][2] - text_size[0], cone[0][3] + text_size[1])
        cv2.putText(img, note, text_position, font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
    if (show):
        cv2.imshow("distances",img)
        cv2.waitKey(0)
    return img

def process_video(input_video_path, output_video_path, model):
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    i = 0
    while True:
        print(f"Frame {i}")
        i+=1
        ret, frame = cap.read()

        if not ret:
            break

        # Apply find_distances and display functions
        cones = find_distances(frame, model)
        processed_frame = display(cones, frame,False)
        
        # Write the processed frame to the output video
        out.write(processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer objects
    cap.release()
    out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

def process_webcam(model):
    cap = cv2.VideoCapture(0)  # 0 corresponds to the default webcam

    while True:
        # Capture one frame
        ret, frame = cap.read()

        if not ret:
            break

        # Apply find_distances and display functions
        cones = find_distances(frame, model)
        processed_frame = display(cones, frame, False)

        # Display the processed frame
        cv2.imshow("Live Results", processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Pause for 1 second
        time.sleep(0.05)

    # Release video capture object
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

# import model
model_path = "/Users/william/Downloads/fa23_cone_segmentation.pt"
model = YOLO(model_path)
# Run process on webcam
process_webcam(model)

# Specify input and output video paths
#input_video_path = "/Users/william/Downloads/IMG_2163.MOV"
#output_video_path = "/Users/william/Downloads/FILTERED_IMG_2163 .MOV"

# Run process on video
#process_video(input_video_path, output_video_path, model)

# Testing on single image
# imgpath = "/Users/william/Desktop/Screenshot 2024-02-28 at 2.41.06 PM.png"
# img = cv2.imread(imgpath)
# cones = find_distances(img,model)
# display(cones,img,True)




