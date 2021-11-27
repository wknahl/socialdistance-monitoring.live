from base64 import b64encode
from io import BytesIO

import cv2
import numpy as np
from flask import render_template, Response, flash
from werkzeug.exceptions import abort
from app.main import main_bp
from app.main.camera import Camera
from source.video_detector import detect_violation
from source.video_detector import print_notif
import winsound

import json
import time

sub = cv2.createBackgroundSubtractorMOG2()

@main_bp.route("/")
def home_page():
    return render_template("live_detection.html")


@main_bp.route("/live")
def live_detect():
  return render_template("live_detection.html")

@main_bp.route("/video")
def video_detect():
  return render_template("video_detection.html")


def gen(camera):
    while True:
        frame = camera.get_frame()
        frame_processed = detect_violation(frame)
        frame_processed = cv2.imencode('.jpg', frame_processed)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_processed + b'\r\n')

def genn():
    cap = cv2.VideoCapture('videotester2.avi')

    while(cap.isOpened()):
      ret, frame = cap.read()
      if ret:
        frame = detect_violation(frame)
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
      key = cv2.waitKey(20)
      if key == 27:
            break


@main_bp.route('/video_feed')
def video_feed():
    return Response(gen(
        Camera()
    ),
        mimetype='multipart/x-mixed-replace; boundary=frame')

#stream local files
@main_bp.route("/streamread")
def streamread():
  return Response(genn(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def allowed_file(filename):
    ext = filename.split(".")[-1]
    is_good = ext in ["jpg", "jpeg", "png"]
    return is_good

@main_bp.route("/listen")
def listen():

  def respond_to_client():
    while True:
      global counter
      global statusNotif
      statusNotif = print_notif()
    #   print(statusNotif)
      color = "white"
    #   with open("color.txt", "r") as f:
    #     color = f.read()
    #     print("**")
      if(color == "white"):
        # print(counter)
        # counter += 1
        _data = json.dumps({"color":color, "counter":statusNotif})
        yield f"id: 1\ndata: {_data}\nevent: online\n\n"
      time.sleep(0.5)
  return Response(respond_to_client(), mimetype='text/event-stream')



      




        


