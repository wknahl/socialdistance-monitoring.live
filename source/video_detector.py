# USAGE
# python social_distance_detector.py --input testvideo1.mp4
# python social_distance_detector.py --input pedestrians.mp4 --output output.avi

# import the necessary packages
from typing import Text
from models import social_distancing_config as config
from models.detection import detect_people
from scipy.spatial import distance as dist
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import cv2
import os
import winsound

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

notification = 0

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
#vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
def video_detect():
    #vs = VideoStream(src=0).start()
    vs = cv2.VideoCapture('main/videotester.mp4')
    # loop over the frames from the video stream
    while True:
        frame = vs.read()
        frame = detect_violation(frame)

def detect_violation(frame):
    while True:
        # resize the frame and then detect people (and only people) in it
        frame = imutils.resize(frame, width=900)
        results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
        
        # deklarasi notif
        global notification

        # inisialisasi kumpulan indeks yang melanggar jarak sosial minimum
        violate = set()

        # inisialisasi indexs notif
        notification = 0

        dur = 100

        # memastikan setidaknya terdapat 2 orang yang terdeteksi
        if len(results) >= 2:
           
            # mengestrak semua centroid dari hasil perhitungan jarak euclidean
            # antara semua pasangan centroid
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # looping dari matriks 
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):

                    # periksa untuk melihat apakah jarak antara 
                    # dua pasangan centroid kurang dari jumlah piksel yang dikonfigurasi
                    if D[i, j] < config.MIN_DISTANCE:
                        # memperbarui indeks pasangan centroid
                        violate.add(i)
                        violate.add(j)

        # looping result
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # ekstrak BB dan koordinat centroid, lalu inisialisasi warna anotasi
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            # update warna indeks pelanggaran
            if i in violate:
                color = (0, 0, 255)
                notification = 1
                winsound.PlaySound("warning.wav", winsound.SND_FILENAME)

            # menggambar (1) kotak pembatas di sekitar orang tersebut 
            # dan (2) koordinat centroid orang tersebut
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)
        
        # memberikan informasi jumlah total pelanggaran jarak sosial pada frame
        text = "Violations: {}".format(len(violate))
        #sound = winsound.PlaySound("warning", winsound.SND_FILENAME)
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2),
        

        return frame


	        
def print_notif():
    global notification
    # print(statusNotif)
    if notification == 1:
        # print(notification)
        return notification
    return notification
        

if __name__ == '__main__':
    video_detect()



	# # if an output video file path has been supplied and the video
	# # writer has not been initialized, do so now
	# if args["output"] != "" and writer is None:
	# 	# initialize our video writer
	# 	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	# 	writer = cv2.VideoWriter(args["output"], fourcc, 25,
	# 		(frame.shape[1], frame.shape[0]), True)

	# # if the video writer is not None, write the frame to the output
	# # video file
	# if writer is not None:
	# 	writer.write(frame)
