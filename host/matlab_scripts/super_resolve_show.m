%% Script to show LR and SR images side by side.

imnum = 3;
% Load the data
load(sprintf('../tmp/im_LR%d.mat',imnum));
load(sprintf('../tmp/im_SR%d.mat',imnum));

% Load first frame
im_lr = im_cell(:,:,1);
im_sr = IM1m(:,:,1);

% Scale the LR image
im_scaled = imresize(im_lr, 2.0, 'bicubic');

% Concatenate both images.
im_cat = [im_scaled im_sr];

% Show the image
figure(); imshow(uint8(im_cat));

% Save the image
imwrite(uint8(im_cat), sprintf('../tmp/imcat%d.png', imnum));
imwrite(uint8(im_scaled), sprintf('../tmp/imlr%d.png', imnum));
imwrite(uint8(im_sr), sprintf('../tmp/imsr%d.png', imnum));