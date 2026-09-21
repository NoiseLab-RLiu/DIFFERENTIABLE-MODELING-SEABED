function [cVec, d] = computeCAndD(a,b)

N = length(a)-1;
M = length(b)-1;

if a(end)~=1
	error('The highest-order coefficient of A(s) must be 1 (a_N=1)')
end

if M<N
	error('M<N => feedthrough not required. This function handles M=N')
elseif M>N
	error('M>N => not a proper transfer function.')
end

d = b(end);

cVec = zeros(N,1);

for k=1:N
	cVec(k) = b(k) - d*a(k);
end
end