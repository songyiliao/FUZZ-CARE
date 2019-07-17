function [DATA,R] = FUZZ_CARE(data,para)

TrN = para.TrN; 
Kmax = para.K;
window = para.w;
para_kernel = para.para_kernel;
para_MU = para.Model_U;
para_WU = para.Window_U;
N_test = para.N_test;
alpha = para.alpha;
lambda1 = para.lambda1;
lambda2 = para.lambda2;

%-------------------------preprocessing -----------------------------------
if isempty( isnan(data) )~=1
    [~,m]=size(data);
    for i = 1:m
        data(isnan(data(:,i)),i) = mean(data(:,i),'omitnan');
    end
    n=[];    m=[];
end
[trainset,testset] = construct_online(data,TrN);
in_n = size(trainset.input,2);
[TRN,TES]=DataNormalization(trainset.input,testset.input,trainset.output,testset.output,TrN);

%-----------------------intialization --------------------------
for k = 2:Kmax
    if para_kernel == 0
        [DATAK{k}.theta, DATAK{k}.U, DATAK{k}.center, DATAK{k}.obj_fcn] = FUZZCARE_fcm(TRN.input,TRN.output, k, [alpha;lambda2;lambda1]);
    elseif para_kernel == 1
        [DATAK{k}.theta, DATAK{k}.U, DATAK{k}.center, DATAK{k}.obj_fcn] = FUZZCARE_kernelfcm(TRN.input,TRN.output, k, [alpha;lambda2;lambda1]);
    else 
        print('para_kernel is limited to 0 or 1')
    end
    InsampleError = testinsample(DATAK{k}.theta,TRN.input,trainset.output,DATAK{k}.U,TRN.outputn);
    KError(k) = mean(InsampleError);
end
[~,Kmin] = min(KError(2:end));K = Kmin+1;
DATA.theta = DATAK{K}.theta;DATA.U = DATAK{K}.U;
DATA.center = DATAK{K}.center;DATA.obj_fcn = DATAK{K}.obj_fcn;ntime = 1;
% store the intial predictors
if isempty(N_test)
    N_test = length(testset.output);
end
u = DATA.U(end,:);DATA.u{1,:} = u;

for i = 1:N_test
    X = [1 TES.input(i,:)];
    DATA.yvaluex(i) = sum( u.*(X*DATA.theta) ,2);
%------------------ update U by fcm or kernel-fcm----------------------
    cluster_data = [TES.input(i,:) TES.output(i)];
    norm_XC = sum((repmat(cluster_data,K,1) - DATA.center).^2,2);  %cluster_n*1
    if para_kernel == 1%------------compute kernelnorm and membership----------
        Kernel_x = ones(K,1)*cluster_data; Kernel_y = DATA.center;
        Kernelnorm_XC = KernelFunction(Kernel_x,Kernel_x)+...
            KernelFunction(Kernel_y,Kernel_y)-2*KernelFunction(Kernel_x,Kernel_y);
        norm_XC = Kernelnorm_XC;
    end

    if para_MU == 1
        for k = 1:K
            U_kx = DATA.U(:,k)*ones(1,in_n+1).*X;    % generate U_kx
            DATA.theta(:,k) = DATA.theta(:,k)+2*alpha/(TrN)*(U_kx'*(TES.output(i)-sum(U_kx*DATA.theta,2))+lambda2*DATA.theta(:,k));
        end
    end
    u = 1./sum(norm_XC*ones(1,K)./(ones(K,1)*norm_XC'),2); %update swift        
    u=u';DATA.U(TrN+i,:) = u;
    TRN.input = [TRN.input;TES.input(i,:)]; TRN.output = [TRN.output;TES.output(i)];

if para_WU ==1
    if i>=ntime*window
        for k = 2:Kmax
            if para_kernel==0
                [DATAK{k}.theta, DATAK{k}.U, DATAK{k}.center, DATAK{k}.obj_fcn] = FUZZCARE_fcm(TRN.input,TRN.output, k,[alpha;lambda2;lambda1]);
            elseif para_kernel == 1
                [DATAK{k}.theta, DATAK{k}.U, DATAK{k}.center, DATAK{k}.obj_fcn] = FUZZCARE_kernelfcm(TRN.input,TRN.output, k,[alpha;lambda2;lambda1]);
            else
                print('para_kernel is limited to 0 or 1')
            end
            InsampleError = testinsample(DATAK{k}.theta,TRN.input,TRN.output,DATAK{k}.U);
            KError(k) = mean(InsampleError);
        end
        [~,Kmin] = min(KError(2:end));K = Kmin+1;
        u = DATAK{K}.U(end,:);DATA.theta = DATAK{K}.theta; DATA.center = DATAK{K}.center;DATA.obj_fcn = DATAK{K}.obj_fcn;
        DATA.U = DATAK{K}.U;
        ntime = ntime+1;
    end% 
end
%----------------------------update fcm------------------------------------
end
DATA.yvalue = mapminmax('reverse',DATA.yvaluex,TRN.outputn);
DATA.output = testset.output(1:N_test);
if size(DATA.yvalue) ~= size(DATA.output)
    DATA.yvalue = DATA.yvalue';
end
R = CompMetric(DATA.yvalue,DATA.output,length(testset.output));
end

%---------------Normalization----------------------------------------------
function [TRN,TES]=DataNormalization(trainsetinput,testsetinput,trainsetoutput,testsetoutput,TrN)
    [TRN.inputx,TRN.inputn] = mapminmax([trainsetinput;testsetinput]');
    [TRN.outputx,TRN.outputn] = mapminmax([trainsetoutput;testsetoutput]');
    TES.inputx = mapminmax('apply',testsetinput',TRN.inputn);TES.outputx = mapminmax('apply',testsetoutput',TRN.outputn);
    TRN.input = TRN.inputx';TRN.output = TRN.outputx';TES.input = TES.inputx';TES.output = TES.outputx';
    TRN.input(TrN+1:end,:) = [];TRN.output(TrN+1:end,:) = [];
end