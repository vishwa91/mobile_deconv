function V = tv(DU,epsilon)
%
% V = TV(DU,epsilon)
%
% support function for Total Variation regularization
%
% For potential function phi(s) = sqrt(s^2)
% this function returns phi(s)'/s = 1/sqrt(s^2)
% i.e.
% 1./sqrt(|grad(U)|^2)
% 
% if |grad(U)| < epsilon then return 1/epsilon

V = max(min(sqrt(DU),1/epsilon),epsilon);
V = 1./V;
