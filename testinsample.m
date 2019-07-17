function insampleError = testinsample(theta,input,y,U,outputn)
for i = 1:length(input)
    X =  [1 input(i,:)];
    u = U(i,:);
    yvaluex(i,1) = sum( u.*(X*theta) ,2);    
end
if  nargin >4
    yvalue = mapminmax('reverse',yvaluex,outputn);
else
    yvalue = yvaluex;
end
insampleError = abs(yvalue-y);
    