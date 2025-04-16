from licensePlateDetection.pipeline.training_pipeline import TrainPipeline
import sys
import time
import os
from licensePlateDetection.pipeline.training_pipeline import TrainPipeline
from licensePlateDetection.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from flask_cors import CORS, cross_origin
from licensePlateDetection.constant.application import APP_HOST, APP_PORT
from licensePlateDetection.Database.anpr_database import ANPD_DB
from licensePlateDetection.Api.Api import Api_req
from licensePlateDetection.Ocr.Ocr import ocr_detection
import shutil
import re
import io,base64
import json
from bson import json_util
import requests
import google.generativeai as genai
import numpy as np
from PIL import Image, ImageEnhance
import torch
import cv2
import subprocess
import uuid
import traceback
app = Flask(__name__, static_folder="../frontend/dist", static_url_path="/")
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

# Deployment part for vercel which was used to serve the frontend on port 8000(backend)
@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")



# training model 
@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training Successfull!!"

# using image
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        data = request.get_json()
        image_base64 = data.get("image")

        if not image_base64:
            return Response("No image provided in request", status=400)

        # Save decoded image
        decodeImage(image_base64, clApp.filename)

        # Run YOLOv5 detection using subprocess
        subprocess.run([
            "python", "yolov5/detect.py",
            "--weights", "yolov5/best.pt",
            "--img", "416",
            "--conf", "0.5",
            "--source", "data/inputImage.jpg",
            "--save-txt",
            "--save-conf"
        ], check=True)

        # Dynamically find the latest exp folder
        result_dir = "yolov5/runs/detect"
        latest_exp = sorted(
            [d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))],
            key=lambda x: os.path.getctime(os.path.join(result_dir, x))
        )[-1]

        result_path = os.path.join(result_dir, latest_exp)
        result_image_path = os.path.join(result_path, "inputImage.jpg")
        bbox_path = os.path.join(result_path, "labels", "inputImage.txt")

        # Load the image
        image = Image.open(result_image_path)

        # Read the bounding box coordinates
        with open(bbox_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.split()
            x_center, y_center, width, height = map(float, parts[1:5])

            img_width, img_height = image.size
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_image = cropped_image.resize((720, 360))

            # Enhance image for OCR
            cropped_image = ImageEnhance.Sharpness(cropped_image).enhance(2.0)
            cropped_image = ImageEnhance.Contrast(cropped_image).enhance(1.5)

            cropped_image_path = os.path.join(result_path, "crop.jpg")
            cropped_image.save(cropped_image_path)

        # OCR processing
        print("Extracting text from cropped image...")
        text = ocr_detection().extracting_text(cropped_image)

        # Encode cropped image to base64
        opencodedbase64 = encodeImageIntoBase64(cropped_image_path)
        result = {"image": opencodedbase64.decode('utf-8')}
        shutil.rmtree("yolov5/runs")

        # Connect to DB and search for vehicle
        dbS = ANPD_DB("ANPD", "anpr_data")
        vechile_data = dbS.get_vehicle_by_registration_number(text)

        if vechile_data:
            print(f"{text} number plate found in database")
            reg_data = json.loads(json_util.dumps(vechile_data))
        else:
            print("Not found in DB, fetching from API...")
            res_data = Api_req().fetchApi(text)

            with open('data.json', 'w') as json_file:
                json.dump(res_data, json_file, indent=4)

            dbS.insert_data("data.json")
            os.remove("data.json")

            reg_data = json.loads(json_util.dumps(
                dbS.get_vehicle_by_registration_number(text)
            ))

        response = {
            "processed_image": result,
            "reg_data": reg_data
        }

        return jsonify(response)

    except subprocess.CalledProcessError as e:
        traceback.print_exc()
        return Response("Error running detection script", status=500)

    except Exception as e:
        traceback.print_exc()
        if os.path.exists("yolov5/runs"):
            shutil.rmtree("yolov5/runs", ignore_errors=True)
        return Response(f"Internal server error: {str(e)}", status=500)

   


# Using text
@app.route("/text", methods=['POST'])
@cross_origin()
def predictText():
    try:

        # CHECK IN DATABASE IS GIVEN PLATE EXSIST IF YES THEN WE RETURN DETAIL DIRECTLY FROM DATA BASE 
        # ELSE FIRST WE GET DATA AND THEN STORE IN DATABASE THEN GET IT
        data = request.get_json()
        license_plate = data.get("text")

        dbS = ANPD_DB("ANPD","anpr_data")
        vechile_data  = dbS.get_vehicle_by_registration_number(license_plate)
        if vechile_data:
            print(vechile_data)
            reg_data = json.loads(json_util.dumps(vechile_data))
            response = {
                "reg_data":reg_data
            }
            return jsonify(response)

        else:
            print("fetching from api")

            res_data = Api_req().fetchApi(license_plate)
            with open('data.json', 'w') as json_file:
                json.dump(res_data,json_file,indent=4)
            
            dbS.insert_data("data.json")
            os.remove("data.json")
            vechile_data  = dbS.get_vehicle_by_registration_number(license_plate)
            reg_data = json.loads(json_util.dumps(vechile_data))
            response = {
                "reg_data":reg_data
            }
            print(response)
            return jsonify(response)


    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")

@app.route('/video_feed')
@cross_origin()
def video_feed():
    cap = cv2.VideoCapture(0)
    crop_cascade = cv2.CascadeClassifier('model/haarcascade_russian_plate_number.xml')
    plate_captured = False
    stable_start_time = None
    if not os.path.exists('plates'):
        os.makedirs('plates')

    def generate():
        nonlocal plate_captured, stable_start_time
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                crops = crop_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(crops) > 0 and not plate_captured:
                    (x, y, w, h) = crops[0]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    if stable_start_time is None:
                        stable_start_time = time.time()
                    elif time.time() - stable_start_time >= 5:
                        crop_img = frame[y:y + h, x:x + w]
                        crop_path = os.path.join("plates", "detected_plate.jpg")
                        cv2.imwrite(crop_path, crop_img)

                        plate_captured = True

                        # Process the cropped image
                        cropped_image = Image.open(crop_path).resize((720, 360))
                        cropped_image = ImageEnhance.Sharpness(cropped_image).enhance(2.0)
                        cropped_image = ImageEnhance.Contrast(cropped_image).enhance(1.5)

                        print("Extracting text")
                        text = ocr_detection().extracting_text(cropped_image)
                        print(text)

                        license_plate = text
                        dbS = ANPD_DB("ANPD", "anpr_data")
                        print("Getting data")
                        vechile_data = dbS.get_vehicle_by_registration_number(license_plate)
                        print(vechile_data)

                        if vechile_data:
                            reg_data = json.loads(json_util.dumps(vechile_data))
                        else:
                            print("Fetching from API")
                            res_data = Api_req().fetchApi(license_plate)
                            with open('data.json', 'w') as json_file:
                                json.dump(res_data, json_file, indent=4)
                            dbS.insert_data("data.json")
                            os.remove("data.json")
                            vechile_data = dbS.get_vehicle_by_registration_number(license_plate)
                            reg_data = json.loads(json_util.dumps(vechile_data))

                        response = {
                            "type": "detection",
                            "processed_image": encodeImageIntoBase64(crop_path).decode('utf-8'),
                            "reg_data": reg_data
                        }

                        yield f"data: {json.dumps(response)}\n\n"

                        plate_captured = False
                        stable_start_time = None
                        if os.path.exists(crop_path):
                            os.remove(crop_path)

                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                yield f"data: {{\"type\": \"video\", \"image\": \"{frame_base64}\"}}\n\n"
                time.sleep(0.1)  # Control streaming speed
        except GeneratorExit:
            print("Stream closed.")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cap.release()

    return Response(generate(), mimetype='text/event-stream')

     
if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host=APP_HOST, port=8000)