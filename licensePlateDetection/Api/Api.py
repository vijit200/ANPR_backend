import requests

class Api_req:

    def fetchApi(self,text):
        url = "https://rto-vehicle-information-india.p.rapidapi.com/getVehicleInfo"

        payload = {
                "vehicle_no": text,
                "consent": "Y",
                "consent_text": "I hereby give my consent for Eccentric Labs API to fetch my information"
        }
        headers = {
                "x-rapidapi-key": "5bf971a686msh50f19596d409694p12ce84jsncf9df23061a3",
                "x-rapidapi-host": "rto-vehicle-information-india.p.rapidapi.com",
                "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)

        print(response.json())

        return response.json()