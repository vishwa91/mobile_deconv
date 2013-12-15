clc; clear all; close all;

% Load the estimated TSF here
load TSF_align_hand;
% tx_limit=-3:1:3;
% ty_limit=-3:1:3;
% tz_limit=1;
% rx_limit=0;
% ry_limit=0;
% rz_limit=-2:0.25:2;

tx_limit=-5:1:5;
ty_limit=-5:1:5;
tz_limit=1;
rx_limit=0;
ry_limit=0;
rz_limit=-1:0.25:1;

weight=tsf_cell{7};
tx=tsf_cell{2};
ty=tsf_cell{1};
rz=tsf_cell{6};

[tx_loc,ty_loc,ang_loc, scal_loc]=ndgrid(tx_limit,ty_limit,rz_limit, tz_limit);
tsf=zeros(size(tx_loc));
for i=1:length(weight)
    tsf(find((tx_limit)==tx(i)), find((ty_limit)==ty(i)), find((rz_limit)==rz(i)))=weight(i);
%     find((tz_limit)==tz(i)))=tz(i);
end
    
% save TSF_cornflakes_AB_PRML tsf tx_loc ty_loc ang_loc scal_loc

% figure; imshow(tsf,[]);
