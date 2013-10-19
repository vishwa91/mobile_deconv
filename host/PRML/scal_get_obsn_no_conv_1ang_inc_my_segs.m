% 24TH FEB 2010
% to generate
%
function [blurim] = scal_get_obsn_no_conv_1ang_inc_my_segs(csf,ty_loc,tx_loc,ang_loc,scal_loc,Href,width,height,im1,invrs,d,normal_fn)
load(normal_fn);
% d =800;

im_acc=zeros(height,width);
Hc = [ 1 0 -width/2;
    0 1 -height/2;
    0 0 1 ];
for no_ang=1:length(ang_loc(1,1,:,1))
    ang=ang_loc(1,1,no_ang,1)*pi/180;
    R = [cos(ang) sin(ang) 0;-sin(ang) cos(ang) 0;0 0 1];
    for no_scal = 1:length(scal_loc(1,1,1,:))
        s = scal_loc(1,1,1,no_scal);
        Hs = [ s 0 0;
            0 s 0;
            0 0 1];
        for no_trx=1:length(tx_loc(1,:,1,1))
            for no_try=1:length(ty_loc(:,1,1,1))
                weight=csf(no_try,no_trx,no_ang,no_scal);
                if(weight>0)
                    tx = tx_loc(no_try,no_trx,no_ang,no_scal);
                    ty = ty_loc(no_try,no_trx,no_ang,no_scal);
                    %             tx = tx/d;
                    %             ty = ty/d;
                    Htr =[ 1 0 tx;
                        0 1 ty;
                        0 0 1];
                    if invrs == 0
                        txed_patch = warping(im1,Href*inv(Hc)*(R+(1/d)*[tx ty 0]'*n')*Hc*inv(Href), width, height,'bicubic' );
                    else
                        txed_patch = warping(im1,Href*inv(Hc)*inv(R+(1/d)*[tx ty 0]'*n')*Hc*inv(Href), width, height,'bicubic' );
                    end
                    %                txed_patch(txed_patch == 255) = 0;
                    im_acc=im_acc+txed_patch*weight;
                end
            end
        end
    end
end
blurim = im_acc;
return;
