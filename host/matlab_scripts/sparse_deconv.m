function im_latent = sparse_deconv(tx, ty, imblur, k_size)
    % Sparse deconvolution function. Since we don't have time to port it
    % to python, we port our code to matlab and leverage the existing
    % sparse deconvolution code.
    
    [Y, X] = size(imblur); % Size of the image
    
    % Generate the differentiation matrices.
    [px, py, pxx, pxy, pyy] = gen_partialmat(Y, X);
    
    % Unknown constants
    sigma_s = 2;
    sigma_r = 0.5;
    support_size = 5;
    dt = 1; % For shock filter I guess.
    thresh = truncate_thresh(imblur,k_size,2);
    f_l = 800; % Focal length. Can be arbitrary.
    alpha = 8;
    lambda = 0.1; % TV weight
    
    % Get some prediction matrix. No clue again.
    [Px,Py] = prediction(imblur,sigma_s,sigma_r,support_size,dt,thresh);
    
    % Weight matrix is all ones because we use the direct tx and ty
    weight = ones(length(tx));
    
    % tz is all ones
    tz = ones(length(tx));
    % rx, ry, rz is all zeros because we assume no rotation.
    rx = zeros(length(tx));
    ry = zeros(length(tx));
    rz = zeros(length(tx));
    
    % Construct the 6D kernel. 
    kernel = mybuild_K_mat_meshgrid(imblur,tx,ty, ...
                tz,rx,ry,rz,weight,f_l);
    
    % Now execute the sparse deconvolution. 
    [im_latent] = spv_deconv_partial(kernel, imblur, Px, Py, alpha, ...
                                    k_size, px, py, pxx, pxy, pyy, ...
                                    imblur, lambda);
