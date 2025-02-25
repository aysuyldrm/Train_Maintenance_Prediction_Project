import mysql.connector
import random # only for test and control
import time 
import json
from datetime import datetime
from sensor_data_creation import simulate_train_data

# Configuration
TRAIN_ID = "Train001"
DATA_POINT_PER_MINUTE = 10  # Simulate 10 data per minute
SIMULATION_DURATION_TIME = 3 * 1         #60 * 24 * 7 Simulate a week of data
DATABASE_FILE = "train_data.json" # File to store simulated data (replace with a real database if needed)


# MySQL Configuration

MYSQL_HOST = "your_local_address" # Or your MySQL server address (127.0.0.1)
MYSQL_USER = "your_user_name" # Your MYSQL username (root)
MYSQL_PASSWORD = "your_database_password" # Your MYSQL password (12345)
MYSQL_DATABASE = "your_database_name" # Your database name (train_maintenance)
MYSQL_PORT = 3306 # Default MySQL port


def insert_data_into_mysql(data):
    try:
        mydb = mysql.connector.connect(
            host = MYSQL_HOST,
            user = MYSQL_USER,
            port = MYSQL_PORT,
            password = MYSQL_PASSWORD,
            database = MYSQL_DATABASE
        )
        mycursor = mydb.cursor()

        for data_point in data:
        
            sql = "INSERT INTO sensor_data (train_id, timestamp, engine_temp,\
                                        engine_vibration,engine_rpm,wheel_temp, \
                                        wheel_wear,track_health,brake_pad_wear, \
                                        brake_temp,brake_pressure, \
                                        brake_force,fuel_usage, \
                                        battery_health,humidity, \
                                        air_temp,door_status, \
                                        window_status,latitude, \
                                        longitude,speed, \
                                        emergency_system,weather_temp, \
                                        wind_speed,precipitation, \
                                        diagnostic_data,error_codes, \
                                        cargo_load,cargo_temp,maintenance) VALUES (%s, %s, %s, %s, %s, %s, %s, \
                                                                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
                                                                    %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (data_point['train_id'], data_point['timestamp'], data_point['engine_temp'], \
                        data_point['engine_vibration'], data_point['engine_rpm'], data_point['wheel_temp'], \
                        data_point['wheel_wear'], data_point['track_health'], data_point['brake_pad_wear'], \
                        data_point['brake_temp'], data_point['brake_pressure'], data_point['brake_force'], \
                        data_point['fuel_usage'], data_point['battery_health'], data_point['humidity'], \
                        data_point['air_temp'], data_point['door_status'], data_point['window_status'], \
                        data_point['latitude'], data_point['longitude'], data_point['speed'], \
                        data_point['emergency_system'], data_point['weather_temp'], data_point['wind_speed'], \
                        data_point['precipitation'], data_point['diagnostic_data'], data_point['error_codes'], \
                        data_point['cargo_load'], data_point['cargo_temp'], data_point['maintenance'])
            
            mycursor.execute(sql, val)
            mydb.commit()

            print(f"{len(data)} records inserted into the database.")
    except mysql.connector.Error as err:
        if err.errno == 1062: # Check for duplicate entry error
            print("Duplicate entry error. Some records may not have benn inserted.")
        else:
            print(f"Error: {err}")
    finally:
        if mydb.is_connected():
            mycursor.close()
            mydb.close()

# your simulate_train_data function
simulate_train_data()

if __name__ == "__main__":
    data = simulate_train_data() # Returns the simulated data
    insert_data_into_mysql(data) # Insert data to the database


