function [x, y] = get_position(ax, ay, az, gx, gy, gz, ...
    idx_start, idx_end)
% Construct the position vectors from the given acceleration vectors

% Constants
G = 9.8;    % Acceleration due to gravity
T = 10e-3;  % Sampling time

acx = ax - gx; acy = ay - gy;
accel_x = acx(idx_start:idx_end); % Remove the normalized average
accel_y = acy(idx_start:idx_end);

x = cumsum(cumsum(accel_x))*G*T*T;  % Integrate. 
y = cumsum(cumsum(accel_y))*G*T*T;
