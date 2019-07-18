clear;clc
% load SynData.mat; 
load RealStreams.mat;
for i = 1:8
    data = database(i).Data;
    [DATA{1,i},R{1,i}] = FUZZ_CARE(data,para);
    X = ['Data:',database(i).Name,'  ','MAE:',num2str(mean(R{1,i}.MAE))];
    disp(X)
end 







% Parameters
% para.w = 200; %window size
% para.TrN=2000; % historical data
% para.K=5;
% para.para_kernel = 1;
% para.Model_U = 1;
% para.Window_U = 1;
% para.N_test = [];
% para.alpha = 0.01;
% para.lambda2 = 0.5;
% para.lambda1 = 1;