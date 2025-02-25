'''
 Creation of IoT Devices

    1. Engine Performance Data
        * Temprature
        * Vibration
        * RPM (Revolutions per minute) 
    2. Wheel and Track Condition
        * Wheel Temprature
        * Wheel Gear
        * Track Health
    3. Braking System Data
        * Brake Pad Wear
        * Brake Temprature and Pressure
        * Break Force
    4. Fuel Consumption / Enery Usage
        * Fuel Usage
        * Battery Health
    5. Environmnetal Condition
        * Humidity and Temprature
        * Air Quality
    6. Door and Windows Functionality
        * Door Sensors
        * Window Sensors
    7. Speed and Positioning Data
        * GPS
        * Speed
    8. Passenger and Crew Safety
        * Emergency System
        * Survelliance System
    9. Wheather Data
        * Weather Sensors
    10. Train Diagnosic Health
        * Diagnostic Data
    11. Train Load and Cargo Data
        * Load Sensors
        * Cargo Temprature/Condition
'''


import random
import time
import json
from datetime import datetime


# Confriguration


TRAIN_ID = "Train001"
DATA_POINT_PER_MINUTE = 1000  # Simulate 10 data per minute
SIMULATION_DURATION_TIME = 3 * 1 # 60 * 24 * 7  Simulate a week of data
DATABASE_FILE = "train_data.json" # File to store simulated data (replace with a real database if needed)



# Function to generate random data for each IoT sensor

def generate_sensor_data():
    '''
    Tren bakım tahmini için sentetik bir veri seti oluşturur.

    Öznitelik kolonları, 'maintenance' hedef kolonu ile ilişkili desenlere sahip olacak şekilde üretilir.

    Args:
        None

    Returns:
        data_row (dict): Oluşturulan veri seti.
    '''


    data_row = {}

    # Temel bakım aralığı (ortalama 100 gün civarı)
    maintenance_days = random.randint(90, 110)
    
    # Motor Verileri
    engine_temp = random.uniform(80, 120)
    engine_vibration = random.uniform(0, 5)
    engine_rpm = random.randint(800, 2000)
    
    if engine_temp > 110:
        maintenance_days -= random.randint(5, 15)
    if engine_vibration > 3:
        maintenance_days -= random.randint(3, 10)
    if engine_rpm > 1800:
        maintenance_days -= random.randint(0, 5)
    
    data_row["engine_temp"] = engine_temp
    data_row["engine_vibration"] = engine_vibration
    data_row["engine_rpm"] = engine_rpm


    # Tekerlek Verileri
    wheel_temp = random.uniform(40, 60)
    wheel_wear = random.uniform(0, 100)
    if wheel_temp > 55:
        maintenance_days -= random.randint(2, 8)
    if wheel_wear > 70:
        maintenance_days -= random.randint(5, 20)
    data_row["wheel_temp"] = wheel_temp
    data_row["wheel_wear"] = wheel_wear

    # Ray Sağlığı
    track_health = random.uniform(0, 100)
    if track_health < 30:
        maintenance_days -= random.randint(10, 30)
    data_row["track_health"] = track_health

    # Fren Verileri
    brake_pad_wear = random.uniform(0, 100)
    brake_temp = random.uniform(50, 200)
    brake_pressure = random.uniform(0, 150)
    brake_force = random.uniform(500, 5000)

    if brake_pad_wear > 80:
        maintenance_days -= random.randint(10, 30)
    if brake_temp > 150:
        maintenance_days -= random.randint(5, 15)
    if brake_pressure > 120:
        maintenance_days -= random.randint(2, 8)
    data_row["brake_pad_wear"] = brake_pad_wear
    data_row["brake_temp"] = brake_temp
    data_row["brake_pressure"] = brake_pressure
    data_row["brake_force"] = brake_force

    # Yakıt ve Batarya
    fuel_usage = random.uniform(50, 100)
    battery_health = random.uniform(70, 100)
    if fuel_usage > 90:
        maintenance_days -= random.randint(0, 5)
    if battery_health < 80:
        maintenance_days -= random.randint(3, 10)
    data_row["fuel_usage"] = fuel_usage
    data_row["battery_health"] = battery_health

    # Çevresel Veriler
    humidity = random.uniform(20, 90)
    air_temp = random.uniform(-10, 40)
    weather_temp = random.uniform(-20, 45)
    wind_speed = random.uniform(0, 100)
    precipitation = random.uniform(0, 50)
    if humidity > 80 or humidity < 30:
        maintenance_days -= random.randint(0, 3)
    if air_temp > 35 or air_temp < -5:
        maintenance_days -= random.randint(0, 3)
    if wind_speed > 80:
        maintenance_days -= random.randint(0, 5)
    if precipitation > 30:
        maintenance_days -= random.randint(0, 5)
    data_row["humidity"] = humidity
    data_row["air_temp"] = air_temp
    data_row["weather_temp"] = weather_temp
    data_row["wind_speed"] = wind_speed
    data_row["precipitation"] = precipitation

    # Diğer Durum Verileri
    door_status = random.choice(["open", "closed"])
    window_status = random.choice(["ok", "broken"])
    emergency_system = random.choice(["ok", "triggered"])
    diagnostic_data = random.choice(["ok", "warning", "error"])
    error_codes = random.choice(["E004", "E001", "E002", "E003"])
    if window_status == "broken":
        maintenance_days -= random.randint(0, 2)
    if emergency_system == "triggered":
        maintenance_days -= random.randint(0, 2)
        #maintenance_days = min(maintenance_days, random.randint(0, 2)) # Acil bakım
    if diagnostic_data == "warning":
        maintenance_days -= random.randint(2, 7)
    elif diagnostic_data == "error":
        maintenance_days -= random.randint(7, 20)
        #maintenance_days = min(maintenance_days, random.randint(0, 5)) # Kritik hata, acil bakım


    data_row["door_status"] = door_status
    data_row["window_status"] = window_status
    data_row["emergency_system"] = emergency_system
    data_row["diagnostic_data"] = diagnostic_data
    data_row["error_codes"] = error_codes

    # Konum ve Hız (Konumun bakım ile direkt ilişkisi az, örnek olması için eklenmiştir)
    latitude = random.uniform(-90, 90)
    longitude = random.uniform(-180, 180)
    speed = random.uniform(0, 300)
    data_row["latitude"] = latitude
    data_row["longitude"] = longitude
    data_row["speed"] = speed

    # Kargo Verileri
    cargo_load = random.uniform(10, 50)
    cargo_temp = random.uniform(-10, 25)
    if cargo_load > 40:
        maintenance_days -= random.randint(0, 3)
    if cargo_temp > 20 or cargo_temp < -5:
         maintenance_days -= random.randint(0, 2)
    data_row["cargo_load"] = cargo_load
    data_row["cargo_temp"] = cargo_temp

    # Bakım Günü (Hedef Kolon) - Negatif gün sayısını engelle ve minimum 1 gün yap
    maintenance_days = max(1, maintenance_days)
    data_row["maintenance"] = int(maintenance_days) # Gün sayısı tam sayı olmalı

    return data_row


# Function to simulate data collection and store in the database


def simulate_train_data():
    data = []

    for minute in range(SIMULATION_DURATION_TIME):
        for _ in range(DATA_POINT_PER_MINUTE):
            timestamp = datetime.now().isoformat()  # Use ISO format for timestamp
            sensor_data = generate_sensor_data()
            data_point = {
                'train_id' : TRAIN_ID,
                'timestamp' : timestamp,
                **sensor_data,
            }
            data.append(data_point)
        time.sleep(60/DATA_POINT_PER_MINUTE*6/30)  # Wait for 1 minute
        print(f"Simulated data for minute (minute + 1) of {SIMULATION_DURATION_TIME}") # Print to show progress


        # Store data in JSON file (replace with database connection)

        #with open(DATABASE_FILE, "w") as f:
        #    json.dump(data, f, indent=4) # indent for better readability

        print(f"Simulation complete. Data saved to {0}".format(DATABASE_FILE))




    return data



if __name__ == "__main__":
    simulate_train_data()

        
        