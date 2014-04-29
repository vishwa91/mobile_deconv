% Matlab script to deconvolve images using blind deconvolution function
% but using our constructed psf as the initial estimate. 

% As suggested in the paper by Neel Joshi et al., we don't know the
% scale and the drift. Hence, we keep iterating through a small 
% subspace to correct for it.

clear all;
%% Constants and paths
main_dir = '../output/cam/';
imblur_name = 'saved_im.bmp';
acdat_name = 'saved_ac.dat';
save_dir = '../tmp/matlab_deconv/';

%% Load the data
imblur = double(rgb2gray(imread(strcat(main_dir, imblur_name))));
[acx, acy, acz, gx, gy, gz] = load_accel(strcat(main_dir, acdat_name));
[x_mm, y_mm] = get_position(acx, acy, acz, gx, gy, gz, 1, 21);

%% Main iterations
dist_max = max(hypot(x_mm, y_mm));
count = 0;

% Create weight matrix which will reduce ringing
imweight = edge(imblur, 'sobel', 0.1);
se = strel('disk', 2);
imweight = double(imdilate(imweight, se));
[xdim, ydim] = size(imweight);
%weight = zeros(xdim, ydim, 3);
%weight(:,:,1) = imweight;
%weight(:,:,2) = imweight;
%weight(:,:,3) = imweight;
    
for depth=linspace(2/dist_max, 8/dist_max, 8)
    for xshift=linspace(0, max(abs(x_mm)), 5)
        for yshift=linspace(0, max(abs(y_mm)), 5)
            fprintf('depth = %f at count %d\n', depth, count);
            % Subtract the shifts
            x = x_mm - linspace(0, xshift, length(x_mm));
            y = y_mm - linspace(0, xshift, length(y_mm));

            % Construct the PSF
            psf = construct_kernel(x, y, depth);
            % Start non-blind deconvolution
            %[im, bpsf] = deconvblind(imblur, psf);
            im = deconvreg(imblur, psf);
            % Save the data
            imwrite(uint8(im*20), ...
                strcat(save_dir, sprintf('im%d.bmp',count)));
            %imwrite(uint8(bpsf*300), ...
            %    strcat(save_dir, sprintf('psf%d.bmp',count)));
            count = count + 1;
        end
    end
end


