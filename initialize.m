function [ params ] = initialize(params)
%%  ³õÊ¼»¯ P,Z,B
[n,d] = size(params.X);
params.P = rand(d,params.b);
params.Z = randn(n,params.b);
params.B = sgn(params.Z);
end

