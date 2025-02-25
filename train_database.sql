
-- Create database
CREATE DATABASE IF NOT EXISTS train_maintanance;
USE train_maintanance;

-- Create the train table
CREATE TABLE IF NOT EXISTS train(
    train_id VARCHAR(255) PRIMARY KEY,
    -- Add other train-spesification if needed (model, type, ...)
    model VARCHAR(255),
    type VARCHAR(255)
    );

CREATE TABLE IF NOT EXISTS sensor_data(
    id INT AUTO_INCREMENT PRIMARY KEY,
    train_id VARCHAR(255),
    timestamp DATETIME,
    engine_temp DECIMAL(5,2),
    engine_vibration DECIMAL(5,2),
    engine_rpm INT,
    wheel_temp DECIMAL(5,2),
    wheel_wear DECIMAL(5,2),
    track_health DECIMAL(5,2),
    brake_pad_wear DECIMAL(5,2),
    brake_temp DECIMAL(5,2),
    brake_pressure DECIMAL(5,2),
    brake_force DECIMAL(8,2),
    fuel_usage DECIMAL(6,2),
    battery_health DECIMAL(5,2),
    humidity DECIMAL(5,2),
    air_temp DECIMAL(5,2),
    door_status VARCHAR(20),
    window_status VARCHAR(20),
    latitude DECIMAL(9,6),
    longitude DECIMAL(9,6),
    speed DECIMAL(5,2),
    emergency_system VARCHAR(20),
    weather_temp DECIMAL(5,2),
    wind_speed DECIMAL(5,2),
    precipitation DECIMAL(5,2),
    diagnostic_data VARCHAR(20),
    error_codes VARCHAR(255),
    cargo_load DECIMAL(5,2),
    cargo_temp DECIMAL(5,2),
    maintenance INT(),
    FOREIGN KEY (train_id) REFERENCES train(train_id) -- Foreign key constraint
);

CREATE TABLE IF NOT EXISTS maintenance_events(
    event_id INT AUTO_INCREMENT PRIMARY KEY,
    train_id VARCHAR(255),
    event_type VARCHAR(255), -- "engine_repair", "wheel_replacement"
    start_time DATETIME,
    end_time DATETIME,
    description TEXT,
    FOREIGN KEY (train_id) REFERENCES train(train_id)
);

-- Insert some sample train data

INSERT INTO train(train_id, model, type) VALUES
('Train001', 'ModelX', 'Passenger'),
('Train002', 'ModelY', 'Cargo');



select * from train;