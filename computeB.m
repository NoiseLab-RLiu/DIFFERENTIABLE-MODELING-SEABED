function [b] = computeB(a,c,d)

N = length(a)-1;
M = length(c)-1;

if a(end)~=1
	error('The highest-order coefficient of A(s) must be 1 (a_N=1)')
end
% 
% if M<N
% 	error('M<N => feedthrough not required. This function handles M=N')
% elseif M>N
% 	error('M>N => not a proper transfer function.')
% end


b = zeros(1,N);

for k=1:N
	b(k) = c(k) + d*a(k);
end
b = [b,d];
end