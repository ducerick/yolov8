import cv2
from ultralytics import YOLO
import os
from tqdm import tqdm
import numpy as np


# Load the YOLOv8 model
model = YOLO('yolov8m-seg.pt')

# Open the video file
video_path = "/data1/home/ducbm/ProjectML/yolov8/data/20240330_123733.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize tqdm with the total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pbar = tqdm(total=total_frames)

# Save bounding box path
save_labels = '/data1/home/ducbm/ProjectML/yolov8/predicts/labels'
save_images = '/data1/home/ducbm/ProjectML/yolov8/predicts/images'
save_masks = '/data1/home/ducbm/ProjectML/yolov8/predicts/masks'

# Loop through the video frames
count = 0
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)[0]
        image_out = cv2.imwrite(os.path.join(save_images, str(count) + '.jpg'), results.plot())
        label_out = os.path.join(save_labels, str(count) + '.txt')
        h, w, _ = frame.shape
        with open(label_out, 'w') as f:
            boxes = results.boxes.cpu().numpy()
            cls = boxes.cls
            conf = boxes.conf
            xyxy = boxes.xyxy
            for idx, c in enumerate(cls):
                x_mean = (xyxy[idx][0] + xyxy[idx][3]) / 2
                if c == 0 and conf[idx] > 0.5 and x_mean > w/3 and x_mean < 2*w/3 :
                    bbs = [str(x) for x in list(map(int, xyxy[idx].tolist()))]
                    f.write(" ".join(bbs) + '\n')
                    mask_out = cv2.imwrite(os.path.join(save_masks, str(count) + '.jpg'), (results.masks[idx].data.squeeze().cpu().numpy() * 255).astype('uint8'))
        count += 1
        # # Update the progress bar
        pbar.update(1)
    else:
        # Break the loop if the end of the video is reached
        break

# Close the progress bar
pbar.close()
