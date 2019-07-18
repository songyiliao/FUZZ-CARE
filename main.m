clear;clc
% load SynData.mat; 
load RealStreams.mat;
for i = 1:8
    data = database(i).Data;
    [DATA{1,i},R{1,i}] = FUZZ_CARE(data,para);
    X = ['Data:',database(i).Name,'  ','MAE:',num2str(mean(R{1,i}.MAE))];
    disp(X)
end 
