function [K_mat]=mybuild_K_mat_meshgrid(focussed,tx,ty,tz,rx,ry,rz,weight,f_l)

% focussed=imread('focussed.jpg');
% focussed=im2double(focussed);
% tx=[-2 -1 0 1 2];
% ty=zeros(1,5);
% tz=ones(1,5);
% rx=zeros(1,5);
% ry=zeros(1,5);
% rz=zeros(1,5);
% weight=ones(1,5);
% weight=weight/sum(weight);
% f_l=800;


[Y,X]=size(focussed);
K_mat=sparse(X*Y,X*Y);
[~,H]=myblur6D(focussed,tx,ty,tz,rx*pi/180,ry*pi/180,rz*pi/180,weight,f_l);

[Xgrid,Ygrid]=meshgrid(1:X,1:Y);
Xvec=Xgrid(:)';
Yvec=Ygrid(:)';
onevec=ones(1,length(Xvec));

for hloop=1:length(H)
    if(weight(1,hloop)~=0)
%img_out=zeros(Y,X);
%temp=sparse(X*Y,X*Y);
%temp_inv=sparse(X*Y,X*Y);

homo_mat=H{1,hloop}*[Xvec; Yvec; onevec];
posx=homo_mat(1,:)./homo_mat(3,:);
posy=homo_mat(2,:)./homo_mat(3,:);
dx=posx-floor(posx);
dy=posy-floor(posy);

posyf=floor(posy); posyc=ceil(posy);
posxf=floor(posx); posxc=ceil(posx);

posyf_log=(posyf<=0) | (posyf>Y);
posyc_log=(posyc<=0) | (posyc>Y);
posxf_log=(posxf<=0) | (posxf>X);
posxc_log=(posxc<=0) | (posxc>X);

mult_log=~posyf_log.*~posyc_log.*~posxf_log.*~posxc_log;

posyf(posyf_log)=1;
posyc(posyc_log)=Y;
posxf(posxf_log)=1;
posxc(posxc_log)=X;

p1=sub2ind(size(focussed),posyf,posxf);
p2=sub2ind(size(focussed),posyc,posxf);
p3=sub2ind(size(focussed),posyf,posxc);
p4=sub2ind(size(focussed),posyc,posxc);

row=repmat((1:X*Y)',4,1);
col=[p2'; p3'; p4'; p1'];
value=[(dy.*(1-dx).*mult_log)'; ...
       ((1-dy).*dx.*mult_log)'; ...
       (dy.*dx.*mult_log)'; ...
       ((1-dy).*(1-dx).*mult_log)'];
    

temp=sparse(row,col,value,X*Y,X*Y);

% for i=1:X*Y
%     temp(i,p2(i))=dy(i)*(1-dx(i));
%     temp(i,p3(i))=(1-dy(i))*dx(i);
%     temp(i,p4(i))=dy(i)*dx(i);
%     temp(i,p1(i))=(1-dy(i))*(1-dx(i));
%     temp(i,:)=temp(i,:).*~posyf_log.*~posyc_log.*~posxf_log.*~posxc_log;
% end

% c = H(1,1)*imc + H(1,2)*imr + H(1,3);
% r = H(2,1)*imc + H(2,2)*imr + H(2,3);
% z = H(3,1)*imc + H(3,2)*imr + H(3,3);
% w = 1./z;
% nx = c.*w;
% ny = r.*w;
% 
% 
% counter=0;
% for x=1:X
%     for y=1:Y
%         counter=counter+1;
%         dummy= H{1,hloop}*[x y 1]';
%         posx=dummy(1)/dummy(3);
%         posy=dummy(2)/dummy(3);
%         dx=posx-floor(posx);
%         dy=posy-floor(posy);
%         if(floor(posx)>0 && floor(posy)>0 && ceil(posx)<=X && ceil(posy)<=Y)
%         
% %             img_out(y,x)=(1-dy)*(1-dx)*img(floor(posy),floor(posx)) ...
% %                      +dy*(1-dx)*img(ceil(posy),floor(posx)) ...
% %                      +(1-dy)*dx*img(floor(posy),ceil(posx)) ...
% %                      +dy*dx*img(ceil(posy),ceil(posx));
%            p1=[floor(posy),floor(posx)];
%            p2=[ceil(posy),floor(posx)];
%            p3=[floor(posy),ceil(posx)];
%            p4=[ceil(posy),ceil(posx)];
%            temp(counter,ind2vec(p2,Y,X))=dy*(1-dx);
%            temp(counter,ind2vec(p3,Y,X))=(1-dy)*dx;
%            temp(counter,ind2vec(p4,Y,X))=dy*dx;
%            temp(counter,ind2vec(p1,Y,X))=(1-dy)*(1-dx);
%         end
%     end
% end
K_mat=K_mat+weight(1,hloop)*temp;
clear temp;
%clear temp_inv;
%figure, imshow(uint8(img_out)); 
    end
end
