function [L] = spv_deconv_partial(K_mat,B,Px,Py,alpha,k_size,px_mat,py_mat,pxx_mat,pxy_mat,pyy_mat,L_old,lambda)
% f = (KL-B)'(KL-B) + alpha*(px_mat*L-Px)'(px_mat*L-Px) +alpha*(py_mat*L-Py)'(py_mat*L-Py)
% \partial f = 2*(K'K+alpha*px_mat'*px_mat+alpha*py_mat'*py_mat)*L - 2*(K'B+alpha*px_mat'*Px+alpha*py_mat'*Py)
% KTK:  K'K
% KTB:  K'B


% ---------- Argument defaults ----------
if ~exist('tol','var') || isempty(tol) tol = 1e-1; end;
if ~exist('maxit','var') || isempty(maxit) maxit = 50; end;

% ---------- generate partial matrix
[im_row,im_col] = size(B);
% [px_mat,py_mat] = gen_partialmat(im_row,im_col);

%% iteratively estimate secondary parameter and L
% sigma_s = 2;sigma_r = 0.5;
% support_size = 5;
% dt=1;
% sigma = [sigma_s,sigma_r];
% w = floor(support_size/2);
% L1 = bfilter2(B,w,sigma);
% iter = 30;
% L2 = shock(L1,iter,dt,1,'org');
% [Px,Py] = gradient(L2);

% sigma_s = 2;sigma_r = 0.5;
% support_size = 5;
% dt = 1;
% thresh = truncate_thresh(B,k_size,2);
% [Px,Py] = prediction(B,sigma_s,sigma_r,support_size,dt,thresh);


% A = 2*(K_mat'*K_mat+alpha*px_mat'*px_mat+alpha*py_mat'*py_mat);
% b = 2*(K_mat'*B(:)+alpha*px_mat'*Px(:)+alpha*py_mat'*Py(:))

omega = [50,25,12.5];
temp = K_mat'*K_mat;
K_mat_t = K_mat';
px_mat_t = px_mat';
py_mat_t = py_mat';
pxx_mat_t = pxx_mat';
pxy_mat_t = pxy_mat';
pyy_mat_t = pyy_mat';

A = 2*(omega(1)*temp+omega(2)*px_mat_t*temp*px_mat+omega(2)*py_mat_t*temp*py_mat+...
    omega(3)*pxx_mat_t*temp*pxx_mat+omega(3)*pxy_mat_t*temp*pxy_mat+omega(3)*pyy_mat_t*temp*pyy_mat+...
    alpha*px_mat_t*px_mat+alpha*py_mat_t*py_mat);
clear temp;
b = 2*(omega(1)*K_mat'+omega(2)*px_mat'*K_mat'*px_mat+omega(2)*py_mat'*K_mat'*py_mat+...
    omega(3)*pxx_mat'*K_mat'*pxx_mat+omega(3)*pxy_mat'*K_mat'*pxy_mat+...
    omega(3)*pyy_mat'*K_mat'*pyy_mat)*B(:);
% b = 2*((omega(1)*K_mat_t+omega(2)*px_mat_t*K_mat_t*px_mat+omega(2)*py_mat_t*K_mat_t*py_mat+...
%     omega(3)*pxx_mat_t*K_mat_t*pxx_mat+omega(3)*pxy_mat_t*K_mat_t*pxy_mat+...
%     omega(3)*pyy_mat_t*K_mat_t*pyy_mat)*B(:)+alpha*px_mat_t*Px(:)+alpha*py_mat_t*Py(:));
clear K_mat px_mat py_mat pxx_temp pxy_temp pyy_temp;
clear K_mat_t px_mat_t py_mat_t pxx_temp_t pxy_temp_t pyy_temp_t;

% TV prior implemented now
Potfce='tv';
epsilon=1;

% Note: up_origLR is now being used for TV prior computation
L=-2*lambda*tv_prior(L_old,epsilon,Potfce);

A=A+L;



% ---------- Initialize -------------------------
k = B(:);				           % Current iterate

r = A*k - b;
p = -r;                        % Current conjugate
iter = 0;

% ---------- Begin iteration ------------------------
while ((norm(r) > tol) && (iter < maxit))
  
    rr = r' * r;
    pAp = p' * A * p;
    alpha = rr / (pAp+0.001);    % Compute the step length
    
    k = k + alpha * p;       % Update x
     
    r = r + alpha * A * p;   % Update residual 
    
    rr_new = r' * r;
    rho = rr_new / (rr+0.001);  % Compute beta
    
    p = rho * p - r;        % Update cg vector 
    iter = iter +1;
end

k(k>1) =1;
k(k<0) =0;
L = col2im(k,[1,1],size(B));


