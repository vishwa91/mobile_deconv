function kernel = construct_kernel(x_mm, y_mm, depth)
% Construct the 2D blur kernel given the shifts and depth

x = depth * x_mm;
y = depth * y_mm;

xmax = uint8(max(abs(x)));
ymax = uint8(max(abs(y)));

% Create an empty kernel
kernel = zeros(2*xmax + 3, 2*ymax + 3);

for i=1:1:length(x)
    kernel(uint8(xmax + x(i))+1, uint8(ymax - y(i))+1) = ...
        kernel(uint8(xmax + x(i))+1, uint8(ymax - y(i))+1) + 1;
end