%define the kernel function
function K = KernelFunction(x,y,sigma) %x,y are vectors

if nargin<3
    sigma = 0.01;
end

if min(sigma) ~=0
    K = sum(exp(-0.5/sigma.*(x-y).^2),2 );
else
    K = sum(exp(-0.5*(x-y).^2),2 );
end

end
