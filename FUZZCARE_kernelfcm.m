function [theta, U, center, obj_fcn] = FUZZCARE_kernelfcm(X,y, cluster_n, options)
%
if nargin ~= 3 & nargin ~= 4
	error('Too many or too few input arguments!');
end

data = [X y];

data_n = size(data, 1);   
in_n = size(X, 2);      
if data_n<in_n
    data = data';
    data_n = size(data, 1);
    in_n = size(data, 2);
end


default_options = [0.01;   % learning ratio
        0.5;  % lambda2
        1;	% lambda
		100;	% max. number of iteration
		1e-5;	% min. amount of improvement
		0];      % display info or not 

if nargin == 3
	options = default_options;
else
	% If "options" is not fully specified, pad it with default values.
	if length(options) < 5
		tmp = default_options;
		tmp(1:length(options)) = options;
		options = tmp;
	end
	% If some entries of "options" are nan's, replace them with defaults.
	nan_index = find(isnan(options)==1);
	options(nan_index) = default_options(nan_index);
end
alpha = options(1);
lambda2 = options(2);
lambda = options(3);% Lambda1
max_iter = options(4);		% Max. iteration
min_impro = options(5);		% Min. improvement
display = options(6);		% Display info or not

obj_fcn = zeros(max_iter, 2);	
U = initfcm( cluster_n,data_n); U=U';			

% Main loop
X = [ones(data_n,1) X];                      
center = zeros(cluster_n, in_n+1);            
theta = zeros(in_n+1,cluster_n);              
CostIncreaseIndex = 0;
for o = 1:max_iter

    mf = U'.^2;       
    for k = 1:cluster_n
        center(k,:) = mf(k,:).*(KernelFunction(data,ones(data_n,1)*center(k,:)))'*data/...
            (mf(k,:)*(KernelFunction(data,ones(data_n,1)*center(k,:))));
    end
    % update theta  (In_n*K)
    theta_y = theta; %NAG initialization
    for k = 1:cluster_n
        U_kx = U(:,k)*ones(1,in_n+1).*X;    % generate U_kx
% -----------------use GD to update theta------------------------
        theta(:,k) = theta(:,k)+2*alpha/(data_n)*(U_kx'*(y-sum(U_kx*theta,2))+lambda2*theta(:,k));
    end

    for k = 1:cluster_n
        S_c(:,k) = zeros(data_n,1);
        Kernel_x = data; Kernel_y = ones(data_n,1)*center(k,:);
        Kernelnorm_XC(:,k) = KernelFunction(Kernel_x,Kernel_x)+...
            KernelFunction(Kernel_y,Kernel_y)-2*KernelFunction(Kernel_x,Kernel_y);
        for j = 1:cluster_n
            if j ~=k
                U_kx = U(:,j)*ones(1,in_n+1).*X;
                S_c(:,k) = S_c(:,k)+2*X*theta(:,k).*U_kx*theta(:,j);
            end
        end
    end
    ita_fz = sum((y.*X*theta-S_c)./( (X*theta).^2 + lambda*Kernelnorm_XC ) ,2) -1;
    ita_fm = sum(0.5./ ((X*theta).^2+lambda*Kernelnorm_XC) ,2);
    ita = ita_fz./ita_fm;
    U_fz = 2*y*ones(1,cluster_n).*(X*theta) - S_c - repmat(ita,1,cluster_n);
    U_fm = 2*( (X*theta).^2 + lambda*Kernelnorm_XC );

    Update_row = find(min(U_fz,[],2)>0);
    U(Update_row,:) = U_fz(Update_row,:)./U_fm(Update_row,:);

    obj_fcn(o,1) = sum(sum((X*theta.*U-y*ones(1,cluster_n)).^2));
%     obj_fcn(o,2) = 0;
    for k = 1:cluster_n
        obj_fcn(o,2) = obj_fcn(o,2)+ sum((mf(k,:))'.* sum((data - repmat(center(k,:),data_n,1)).^2,2));
    end
    if o>=2
        if obj_fcn(o,2)>obj_fcn(o-1,2)
            CostIncreaseIndex = CostIncreaseIndex +1;
            if CostIncreaseIndex>=6
                break
            end
        else
            CostIncreaseIndex = 0;
        end
    end

end
iter_n = o;	
obj_fcn = obj_fcn/data_n;
obj_fcn(iter_n+1:max_iter,:) = [];
end