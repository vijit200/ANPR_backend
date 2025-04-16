import urllib.parse
import json
from pymongo import MongoClient, errors

class ANPD_DB:
    def __init__(self, db_name, collection_name):
        self.connection_string = "mongodb+srv://Tushar:" + urllib.parse.quote("Tushar@1730") + "@anpd.jm8oe.mongodb.net/?retryWrites=true&w=majority&appName=ANPD"
        self.db_name = db_name
        self.collection_name = collection_name
        self.db = None
        self.collection = None

    def connect(self):
        try:
            self.client = MongoClient(self.connection_string)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            print(f"Connected to MongoDB: {self.db_name} -> {self.collection_name}")
            return self.client
        except errors.ConnectionFailure as e:
            print(f"Failed to connect to MongoDB: {e}")
        return None

    def insert_data(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                print(f"Successfully read data from {file_path}")
                vehicle_data = data
            connection = self.connect()
            if not connection:
                print("Connection to MongoDB failed. Cannot insert data.")
                return
            
            existing_record = self.collection.find_one({"registration_number": vehicle_data["data"]["registration_no"]})
            if existing_record:
                print("Vehicle with this registration number already exists.")
                return
            
            selected_data = {
                "registration_number": vehicle_data["data"]["registration_no"],
                "registration_date": vehicle_data["data"]["registration_date"],
                "owner_name": vehicle_data["data"]["owner_name"],
                "fuel_type": vehicle_data["data"]["fuel_type"],
                "registration_authority": vehicle_data["data"]["registration_authority"],
                "model": vehicle_data["data"]["maker_model"],
                "ownership":vehicle_data["data"]["ownership_desc"]
            }
            inserted_id = self.collection.insert_one(selected_data).inserted_id
            print(f"Data inserted with ID: {inserted_id}")
        
        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except KeyError as e:
            print(f"Missing key in vehicle data: {e}")
        except errors.PyMongoError as e:
            print(f"MongoDB operation error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
    def show_vehicle_data(self):
        try:
            vehicles = self.collection.find()
            for vehicle in vehicles:
                print(vehicle)
        except errors.PyMongoError as e:
            print(f"Failed to fetch data from MongoDB: {e}")
            
    def get_vehicle_by_registration_number(self, reg_no):
        try:
            connection = self.connect()
            vehicle = self.collection.find_one({"registration_number": reg_no})
            if vehicle:
                # print(f"Vehicle found: {vehicle}")
                return vehicle
            else:
                print(f"No vehicle found with registration number: {reg_no}")
                return False
        except errors.PyMongoError as e:
            print(f"Failed to fetch data from MongoDB: {e}")
            
    def delete_vehicle_data(self, reg_no):
        try:
            result = self.collection.delete_one({"registration_number": reg_no})
            if result.deleted_count > 0:
                print(f"Deleted vehicle with registration number: {reg_no}")
            else:
                print(f"No vehicle found with registration number: {reg_no}")
        except errors.PyMongoError as e:
            print(f"Failed to delete data from MongoDB: {e}")

    