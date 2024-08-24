# USAGE
# python yolo_video.py --input .\videos\airport.mp4
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import sys

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-o", "--output", help="path to output video")
ap.add_argument("-y", "--yolo", default="yolo-coco", help="base path to YOLO directory (default: yolo-coco)")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# YOLO configuration
labelsPath = "../yolo-coco/coco.names"
weightsPath = "../yolo-coco/yolov3.weights"
configPath = "../yolo-coco/yolov3.cfg"

LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Video processing
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

try:
    prop = cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print(f"[INFO] {total} total frames in video")
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

frame_count = 0
frames = []  # List to store processed frames

# Determine output path
input_filename = os.path.basename(args["input"])
base_name, _ = os.path.splitext(input_filename)
output_filename = f"{base_name}_output.mp4"
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_path = os.path.join(output_folder, output_filename)

while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        print("[INFO] No more frames to process.")
        break

    frame_count += 1
    if frame_count >= total:
        print("[INFO] Processed all frames.")
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > args["confidence"]:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{LABELS[classIDs[i]]}: {confidences[i]:.4f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Initialize video writer
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
        if total > 0:
            elap = (end - start)
            print(f"[INFO] single frame took {elap:.4f} seconds")
            print(f"[INFO] estimated total time to finish: {elap * total:.4f} seconds")
            print("[INFO] Processing the output video......")

    writer.write(frame)
    frames.append(frame)  # Add processed frame to list

print("[INFO] cleaning up...")
writer.release()
vs.release()

# Preview the processed video in fullscreen
cv2.namedWindow("Processed Video", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Processed Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

for frame in frames:
    cv2.imshow("Processed Video", frame)
    # Display each frame for a short duration, adjust if necessary
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
sys.exit(0)
