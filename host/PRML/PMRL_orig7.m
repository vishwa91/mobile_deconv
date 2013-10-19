% Projective motion deblurring code
% Thangamani
% modified by Sahana 14/5/13

clc; clear all; close all;
BlurIm = double(imread('corn_flakes_set2.png'));
load TSF_align_corn_flakes_AB;
tsf = tsf_estiml;
% tsf(tsf<0.005)=0;
% tsf = tsf./(sum(sum(sum(tsf))));
fn = 'corn_flakes_set2';

NoIter = 100;
lamda = 10^(-4);
for S = [4]
    bim_fn = strcat(fn,'.png');
    normal_fn = strcat('normal_seg',num2str(S));
    BlurIm = double(imread(bim_fn));
    I = BlurIm; % intialization
    [ht,wd,c]= size(BlurIm);
    
    for i = 1:NoIter
        tic
        fprintf('Iteration %d \n',i);
        BI = blurredimgener_inc_my_segs(I,tsf,ty_loc,tx_loc,ang_loc,scal_loc,0,1,normal_fn);
        Err   = BlurIm./ BI;
        B_Err = blurredimgener_inc_my_segs(Err,tsf,ty_loc,tx_loc,ang_loc,scal_loc,1,1,normal_fn);
        for n = 1:3
            %         for n = 1
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
        imshow(uint8(I));pause(0.5);
        dbim_fn = strcat(fn,'_',num2str(i),'_deblurred.png');
        imwrite(uint8(I),dbim_fn);
        toc
    end
    clear R
    J = I;    
end
