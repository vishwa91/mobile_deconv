% modified Sahana 14/5/13

% Generates image according to the tsf using original image
% gen_matrices_from_tsf creates an array of matrices from the tsf
% the transformations considered are tx, ty ,scale rz only
function blurim1 = blurredimgener_inc_my_segs(im1,tsf,ty_loc,tx_loc,ang_loc,scal_loc,invrs,d,normal_fn)


width = size(im1,2);
height = size(im1,1);
f_size = [size(im1,1) size(im1,2)];
cx     = f_size(2)/2 - width/2;
cy     = f_size(1)/2 - height/2;

fx = 1;fy = 1;
Href = [fx 0 cx;
    0 fy cy;
    0 0 1 ];

blurim1 = zeros(height,width,1);
for col_ind = 1:3
    blurim1(:,:,col_ind) = scal_get_obsn_no_conv_1ang_inc_my_segs(tsf,ty_loc,tx_loc,ang_loc,scal_loc,Href,width,height,im1(:,:,col_ind),invrs,d,normal_fn);
end
% figure;imshow(uint8(blurim1));
% imwrite(uint8(blurim1),'./gopuramBIMG336tsf.jpg');

