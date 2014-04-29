% Matlab script to deconvolve images using sparse deconvolution function
% using our constructed psf as the estimate. 

% As suggested in the paper by Neel Joshi et al., we don't know the
% scale and the drift. Hence, we keep iterating through a small 
% subspace to correct for it.

clear all;
% Add the scripts path
addpath('deconv_scripts');
%% Constants and paths
main_dir = '../output/cam/';
imblur_name = 'saved_im.bmp';
acdat_name = 'saved_ac.dat';
save_dir = '../tmp/matlab_deconv/';

%% Load the data
imblur = double(rgb2gray(imread(strcat(main_dir, imblur_name))));
[acx, acy, acz, gx, gy, gz] = load_accel(strcat(main_dir, acdat_name));
[y_mm, x_mm] = get_position(acx, acy, acz, gx, gy, gz, 1, 21);

%% Main iterations
dist_max = max(hypot(x_mm, y_mm));
count = 0;

for depth=linspace(2/dist_max, 10/dist_max, 10)
    %for xshift=linspace(0, max(abs(x_mm)), 5)
        %for yshift=linspace(0, max(abs(y_mm)), 5)
            %fprintf('Depth = %f, xshift = %f, yshift = %f\n', depth, ...
            %                    xshift, yshift);
            % Subtract the shifts
            x = x_mm; %- linspace(0, xshift, length(x_mm));
            y = y_mm; %- linspace(0, yshift, length(y_mm));
            
            tx = depth*x; ty = depth*y;
            k_size = uint8(max(hypot(tx, ty)));
            imlatent = sparse_deconv(tx, ty, imblur/255.0, k_size)*20.0;
            imwrite(imlatent, strcat(save_dir, ...
                    sprintf('im%d.bmp', count)));
            count = count + 1;
        %end
    %end
end
