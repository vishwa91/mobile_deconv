function L = tv_prior(color_img,epsilon,Potfce)
fprime = inline([Potfce,'(DU,a)']);
U=color_img;
[usize(1) usize(2) usize(3)] = size(U);
N = prod(usize);
VV = ((U(2:end,:,:)-U(1:end-1,:,:)).^2);
VV = vec([fprime(VV,epsilon); zeros(1,size(VV,2),usize(3))]);

% horizontal derivatives
VH = ((U(:,2:end,:)-U(:,1:end-1,:)).^2);
VH = vec([fprime(VH,epsilon), zeros(size(VH,1),1,usize(3))]);

% diagonal derivatives for higher accuracy
VVH1 = ((U(2:end,2:end,:)-U(1:end-1,1:end-1,:)).^2)/2;
VVH1 = vec([fprime(VVH1,epsilon), zeros(size(VVH1,1),1,usize(3)); ...
  zeros(1,usize(2),usize(3))])./2;
VVH2 = ((U(1:end-1,2:end,:)-U(2:end,1:end-1,:)).^2)/2;
VVH2 = vec([zeros(1,usize(2),usize(3)); fprime(VVH2,epsilon), ...
  zeros(size(VVH2,1),1,usize(3))])./2;

% construct sparse matrix L(v)
L = spdiags(VV,-1,N,N) + spdiags(VH,-usize(1),N,N);
L = L + spdiags(VVH1,-usize(1)-1,N,N) + spdiags(VVH2,-usize(1)+1,N,N);
L = L+L';
L = L - spdiags(sum(L,2),0,N,N);