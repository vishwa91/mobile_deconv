function [ax, ay, az, gx, gy, gz] = load_accel(filename)

% Load the data from the file.
accel_data = importdata(filename);
ax = accel_data(1:end, 1)';
ay = accel_data(1:end, 2)';
az = accel_data(1:end, 3)';
gx = accel_data(1:end, 4)';
gy = accel_data(1:end, 5)';
gz = accel_data(1:end, 6)';
