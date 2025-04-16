from flask import Flask, jsonify
from flask_cors import cross_origin
import cv2
import os
from PIL import Image, ImageEnhance
import json
from bson import json_util
from licensePlateDetection.Database.anpr_database import ANPD_DB
from licensePlateDetection.Api.Api import Api_req
from licensePlateDetection.Ocr.Ocr import ocr_detection
from licensePlateDetection.utils.main_utils import decodeImage, encodeImageIntoBase64
app = Flask(__name__)

@app.route("/live", methods=['GET'])
@cross_origin()
def predictLive():
    try:
        haar_cascade_path = "model/haarcascade_russian_plate_number.xml"
        crop_cascade = cv2.CascadeClassifier(haar_cascade_path)

        # Directory to save detected crops
        save_dir = "plates"
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)  # Open webcam
        cap.set(3, 640)  # width
        cap.set(4, 480)  # height

        if not cap.isOpened():
            return jsonify({"error": "Unable to access the camera"}), 500

        crop_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crops = crop_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in crops:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                crop_img = frame[y:y + h, x:x + w]
                crop_path = os.path.join(save_dir, f"crop_{crop_count}.jpg")
                cv2.imwrite(crop_path, crop_img)
                print(f"Saved: {crop_path}")

                # Process the cropped image
                cropped_image = Image.open(crop_path)
                cropped_image = cropped_image.resize((720, 360))
                enhancer = ImageEnhance.Sharpness(cropped_image)
                cropped_image = enhancer.enhance(2.0)
                enhancer = ImageEnhance.Contrast(cropped_image)
                cropped_image = enhancer.enhance(1.5)

                print("Extracting text")
                text = ocr_detection().extracting_text(cropped_image)
                print(text)

                # Process the OCR results
                license_plate = text
                dbS = ANPD_DB("ANPD", "anpr_data")
                vechile_data = dbS.get_vehicle_by_registration_number(license_plate)

                if vechile_data:
                    print(vechile_data)
                    reg_data = json.loads(json_util.dumps(vechile_data))
                    response = {
                        "processed_image": encodeImageIntoBase64(crop_path).decode('utf-8'),
                        "reg_data": reg_data
                    }
                    print(response)
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
                        "processed_image": encodeImageIntoBase64(crop_path).decode('utf-8'),
                        "reg_data": reg_data
                    }
                    print(response)

            cv2.imshow("Crop Detection", frame)

            # Press 'Q' to quit the camera feed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting detection loop.")
                break

        cap.release()
        cv2.destroyAllWindows()

        return jsonify({"message": "Detection loop ended."})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
