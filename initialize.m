function [ params ] = initialize(params)
%%  ��ʼ�� P,Z,B
[n,d] = size(params.X);
params.P = rand(d,params.b);
params.Z = randn(n,params.b);
params.B = sgn(params.Z);
end

