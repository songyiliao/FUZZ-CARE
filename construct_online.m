function [trainset,testset] = construct_online(data,TrN,D)
data(:,isnan(data(1,:)))=[];
[m,n] = size(data);
if m>n
    data = data';
    [m,n] = size(data);
end
if nargin == 2
	D = ones(1,m-1);
end
i=1;j=1;t=1;
while i <= sum(D)
    inputdata(i,1:n+1-j) = data(t,j:n);
    j=j+1;i=i+1;
    if j>D(t)
        t=t+1;j=1;
    end
end


outputdata = data(m,max(D):n);
trainset.input = inputdata(:,1:TrN);
trainset.output = outputdata(1:TrN);
testset.input = inputdata(:,TrN+1:end);
testset.output = outputdata(:,TrN+1:end);

trainset.input = trainset.input';
trainset.output = trainset.output';
testset.input = testset.input';
testset.output = testset.output';




