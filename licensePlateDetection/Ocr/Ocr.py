import google.generativeai as genai
import os
import re

class ocr_detection:

    def extracting_text(self,cropped_image):
        os.environ["GOOGLE_API_KEY"] = 'AIzaSyBLGiR8vhLf_EY9LpVle2ad_xnmJRzQDBY'
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(
                model_name='gemini-1.5-pro-latest')
        prompt = "Extract license plate number from this image."
        ocr_result = model.generate_content([prompt, cropped_image])
        list = ocr_result.text.split(" ")
        print(ocr_result.text, list)
        pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$'

        # Checking pattern of license plate using regex
        if len(list)<2:
            print("here",list[0])
            if "." in list[0]:
                list[0] = list[0].replace(".","")
                text = list[0]
            else:
                text = list[0]
        elif(len(list)>=2):
            ls = ""
            for i in range(len(list)):
                ls += list[i]
            print("here",ls)
            if "." in ls:
                ls = ls.replace(".","")
                text = ls
            else:
                text = ls
                            
        for i in range(len(text)):
            if (i == 2 or i ==3 or i== 6 or i==7 or i==8 or i==9):
                if text[i] == "O":
                    text = text[:i] + "0" + text[i+1:]
                
        if re.match(pattern,text):
            text = text
        else:
            print("regex not match ",text)
            return False

        print(text)
        return text