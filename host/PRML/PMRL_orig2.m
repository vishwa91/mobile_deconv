% Projective motion deblurring code
% Thangamani
% modified by Sahana 14/5/13

clc; clear all; close all;
BlurIm = double(imread('hand_matte.png'));
BlurIm = BlurIm(21:end-20,21:end-20,1);
load estim_tsf;
fn = 'hand_matte';

NoIter = 50;
lamda = 10^(-4);
for S = [4]
    bim_fn = strcat(fn,'.png');
    normal_fn = strcat('normal_seg',num2str(S));
    BlurIm = double(imread(bim_fn));
    I = BlurIm; % intialization
    [ht,wd,c]= size(BlurIm);
    tic
    for i = 1:NoIter
        fprintf('Segment %d, Iteration %d \n',S,i);
        BI = blurredimgener_inc_my_segs(I,tsf,ty_loc,tx_loc,ang_loc,scal_loc,0,1,normal_fn);
        Err   = BlurIm./ BI;
        B_Err = blurredimgener_inc_my_segs(Err,tsf,ty_loc,tx_loc,ang_loc,scal_loc,1,1,normal_fn);        
%         for n = 1:3
        for n = 1
            [gx, gy] = derivative5(I(:,:,n),'x','y');
            normg = sqrt(gx.^2  +gy.^2);
            normg(normg==0) = 1;
            gx = gx./ normg;
            gy = gy./normg;
            gxx = derivative5(gx,'x');
            gxy = derivative5(gy,'x');
            gyx = derivative5(gx,'y');
            gyy = derivative5(gy,'y');
            R(:,:,n) = gxx+gxy+gyx+gyy;
        end        
        I = (I./(1+ lamda*R)).*B_Err;   
        imshow(I);pause(0.5);
    end
    clear R
    J = I;    
    dbim_fn = strcat(fn,'_deblurred.png');
    imwrite(uint8(I),dbim_fn);
end
