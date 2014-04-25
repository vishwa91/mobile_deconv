function [blurred, homography]=myblur6D(img,txs,tys,tzs,rxs,rys,rzs,weight,fl)
im_acc=zeros(size(img));
[Y,X]=size(img);
Kf=[fl 0 0; 0 fl 0; 0 0 1];
Hc = [ 1 0 -(X/2);
       0 1 -(Y/2);
       0 0 1 ];
no_smp=length(rxs);
count=1;
homography=[];
 for no_t=1:no_smp
            Ht = [1 0 txs(no_t);0 1 tys(no_t);0 0 1];
            Hs = [tzs(no_t) 0 0; 0 tzs(no_t) 0; 0 0 1];
            Zm = Kf*expm([0 -rzs(no_t) rys(no_t);rzs(no_t) 0 -rxs(no_t);-rys(no_t) rxs(no_t) 0])*inv(Kf);
            homo=Ht*inv(Hc)*Zm*Hs*Hc;
            homography{1,no_t}=inv(homo);
            txed_patch = mywarping(img,inv(homo));
            im_acc=im_acc+txed_patch*weight(no_t);
            count=count+1;
 end
blurred = im_acc;
return;