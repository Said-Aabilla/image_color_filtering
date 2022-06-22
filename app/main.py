import base64
from flask import Flask, request, jsonify
import cv2
import numpy as np

import matplotlib.colors as mc

import colorsys


app = Flask(__name__)


@app.get("/")
def home_view():
    return "<h1>Welcome to sentient app</h1>"


@app.post("/identify")
def predict():
    img_data = request.get_json().get("bytes")
    img_color = request.get_json().get("color")
    res = str(get_response_bytes(img_data, img_color))
    res = res[2:-1]
    message = {"response": res}
    return jsonify(message)


def get_response_bytes(bytes_data, img_color):

    jpg_original = base64.b64decode(bytes_data)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    image = cv2.imdecode(jpg_as_np, flags=1)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
   
    # Get lower and upper color range
    lower, upper = get_color_range(img_color)

    # Threshold the HSV image to get only the range color
    mask = cv2.inRange(hsv, lower, upper)

    # Bitwise-AND mask and original image
    output = cv2.bitwise_and(image, image, mask=mask)

    #print pixel percentage of that color
    ratio = cv2.countNonZero(mask)/(hsv.size/3)
    print('pixel percentage:', np.round(ratio*100, 2))

    is_success, im_buf_arr = cv2.imencode("image.jpg", output)
    byte_im = im_buf_arr.tobytes()
    return base64.b64encode(byte_im)

    # jpg_as_np = np.frombuffer(byte_im, dtype=np.uint8)
    # image = cv2.imdecode(jpg_as_np, flags=1)
    # cv2.imshow("Color Detected", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def get_color_range(img_color):

    # Convert the color from bgr to hsv
    bgr_color = np.uint8([[[img_color[2],img_color[1], img_color[0]]]])
    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)

    hsv_color = hsv_color[0][0]
    print(hsv_color)

    # define color range
    lower = np.array([hsv_color[0]-10, 100, 100])
    upper = np.array([hsv_color[0]+10, 255,255])
    print(lower, upper)

    return lower, upper
