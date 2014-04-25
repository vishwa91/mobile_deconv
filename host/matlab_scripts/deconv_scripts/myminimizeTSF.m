function [weight_new]=myminimizeTSF(Px,Py,B,tx,ty,tz,rx,ry,rz,f_l,l1_w)
% The function is used to estimate camera motion given predicted gradient
% maps.
% 
% Required input
% Px,Py: predicted gradient maps
% B: blurry image
% omega: weights for partial derivative
% beta: Tihkonov parameter

omega = [sqrt(25),sqrt(12.5)]; % weight for derivative

%[im_row,im_col] = size(B);
[Bx,By] = gradient(B,1);
[Bxx,Bxy1] = gradient(Bx,1);
[Bxy2,Byy] = gradient(By,1);
Bxy=(Bxy1+Bxy2)/2;

% [Pxx,Pxy1] = gradient(Px,1);
% [Pxy2,Pyy] = gradient(Py,1);
% Pxy = (Pxy1+Pxy2)/2;

% build data for optimization
%n_basis = size(pose_set,2);
%dim_im = size(pose_set{1,1},1);
% dim_im=numel(B);
% A = zeros(patch_size(1)*patch_size(2)*n_loc*5,n_proj);
% b = zeros(patch_size(1)*patch_size(2)*n_loc*5,1);
A = [];
% b = [];
% 
% 
bx_temp = Bx;
by_temp = By;
bxx_temp = Bxx;
bxy_temp = Bxy;
byy_temp = Byy;
b = [omega(1)*bx_temp(:);omega(1)*by_temp(:);omega(2)*bxx_temp(:);omega(2)*bxy_temp(:);omega(2)*byy_temp(:)];
    
% K * \partial L
    for i=1:length(tx)
        
        % IMPORTANT: Only the homography is needed and not the image. 
        % the blurred input is the same size as the focussed image and that is
        % all that matters here.
        
        K_mat=mybuild_K_mat_meshgrid(B,tx(i),ty(i),tz(i),rx(i),ry(i),rz(i),1,f_l);
        
        Lx_basis_temp = K_mat*Px(:);
        Ly_basis_temp = K_mat*Py(:);
        [Lxx,Lxy1] = gradient(reshape(Lx_basis_temp,size(B)),1);
        [Lxy2,Lyy] = gradient(reshape(Ly_basis_temp,size(B)),1);
        Lxy = (Lxy1+Lxy2).*0.5;
        
        Lxx_basis_temp = Lxx(:);
        Lxy_basis_temp = Lxy(:);
        Lyy_basis_temp = Lyy(:);
        temp = [omega(1)*Lx_basis_temp;omega(1)*Ly_basis_temp;omega(2)*Lxx_basis_temp;omega(2)*Lxy_basis_temp;omega(2)*Lyy_basis_temp];
        A = [A,temp];
    end

%     temp = A(1:dim_im,:);
%     ATA = temp'*temp*omega(1);
%     ATb = temp'*Bx(:)*omega(1);
%     temp = A(dim_im+1:dim_im*2,:);
%     ATA = ATA+temp'*temp*omega(1);
%     ATb = ATb+temp'*By(:)*omega(1);
%     temp = A(2*dim_im+1:dim_im*3,:);
%     ATA = ATA+temp'*temp*omega(2);
%     ATb = ATb+temp'*Bxx(:)*omega(2);
%     temp = A(3*dim_im+1:dim_im*4,:);
%     ATA = ATA+temp'*temp*omega(2);
%     ATb = ATb+temp'*Bxy(:)*omega(2);
%     temp = A(4*dim_im+1:dim_im*5,:);
%     ATA = ATA+temp'*temp*omega(2);
%     ATb = ATb+temp'*Byy(:)*omega(2);

%clear bx_temp by_temp bxx_temp bxy_temp byy_temp;

% clear Lx_basis_temp Ly_basis_temp Lxx_basis_temp Lxy_basis_temp Lyy_basis_temp;
% clear temp;

%     beta =1;   
%     H = ATA + beta*eye(size(ATA,1));
%     f = -ATb;
%     L = ones(1,length(weight));
%     k = 1;
%     A = [];
%     b = [];
%     l = zeros(length(weight),1);
%     u = [];
%     
%     [w,err,lm] = qpip(H,f,L,k,A,b,l,u,0,0,0);
% w(w<max(w(:))*0.01)=0;
% sum(w)

opts.rsL2 = 0.0;

[w,funVal]=nnLeastR(A,b,l1_w,opts);

weight_new = w./sum(w(:));
%weight_new=w;
end
