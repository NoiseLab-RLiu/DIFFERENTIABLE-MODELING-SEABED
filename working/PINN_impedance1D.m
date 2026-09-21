L = 1;
Nx = 101;
dx = L/(Nx-1);
dt = 0.005;
Nt = 700;
c = 1.6;

x = linspace(0,L,Nx)';
p = zeros(Nx,1);
u = zeros(Nx,1);

p = exp(-((x-0.5)/0.3).^2);

N = 2;
M = 2;
a = [0.9,2.3,1]; %[a1, a0, aN] with aN=1
b = [3,1.5,1.2]; % b = [3,1.5,0.6]

[c_vec, d] = computeCAndD(a,b);

A = zeros(N,N);
A(1:end-1,2:end) = eye(N-1);
A(end, :) = -a(end-1:-1:1);

b_vec = zeros(N,1);
b_vec(end) = 1;

q = zeros(N,1);

coeff_u = c*dt/dx;

P = zeros(Nx,Nt);
U = zeros(Nx,Nt);
lpf_coeffs = -[-0.003, 0.01872, -0.0592, 0.1238, -0.1878, 0.2150-1, -0.1878, 0.1238, -0.0592, 0.01872, -0.003];

fs = 1
nfft = 1024
[H,w] = freqz(lpf_coeffs, 1, nfft, fs)

figure
plot(w,20*log10(abs(H)))

for t=1:Nt
	p(2:end-1) = p(2:end-1) - coeff_u*(u(3:end)-u(1:end-2))/2;
	u(2:end-1) = u(2:end-1) - coeff_u*(p(3:end)-p(1:end-2))/2;
	% Reflection BC
	u(1) = -u(2)
	p(1) = p(2)
	% Impedance BC
	dudx = (u(end)-u(end-1))/dx;
	dpdt = -c^2*dudx;
	p(end) = p(end)+dt*dpdt;
	q_dot = A*q + b_vec*p(end);
	q = q + dt*q_dot;
	u(end) = c_vec'*q+d*p(end);
	P(:,t) = p;
	U(:,t) = u;
	
	p = filtfilt(lpf_coeffs,1,p);
	u = filtfilt(lpf_coeffs,1,u);
end

figure
plot(P(:,150))

figure
imagesc(P)
xlabel('dt','Interpreter','latex')
ylabel('dx','Interpreter','latex')
% Adjust the axes properties
ax = gca;
ax.YTick = [1 20 40 60 80 100]; 
ax.YTickLabel = {'0', '20', '40', '60', '80', '100'}; 
ax.FontSize = 14;                % Change this number as you like
ax.TickLabelInterpreter = 'latex';
% Adjust the colorbar properties
cb = colorbar;
cb.FontSize = 14;                % Same font size as the axes for consistency
cb.TickLabelInterpreter = 'latex';
set(gca, 'YDir', 'normal');
exportgraphics(ax, 'Field_Imp.jpg');
%%
a_ext = [];
b_ext = [];
A_poly=[2.3,0.9,1];
B_poly = [4.68,-0.18,1.2];

degA = length(A_poly)-1;
degB = length(B_poly)-1;

f_min = 0;
f_max = 5; %2000
Npoints = 100; %1000
f = linspace(f_min, f_max, Npoints);
omega = 2*pi*f;

A_poly_for_polyval = fliplr(A_poly);
B_poly_for_polyval = fliplr(B_poly);

a_poly_pinn = [2.325395107269287, 0.8847050666809082, 1];
c_poly_pinn = [1.8959134817123413, -1.2678048610687256];
d_pinn = 1.1952404975891113;
b_poly_pinn = computeB(a_poly_pinn, c_poly_pinn, d);

A_poly_for_polyval_pinn = fliplr(a_poly_pinn);
B_poly_for_polyval_pinn = fliplr(b_poly_pinn);

Z_jw = zeros(1,Npoints);
Z_jw_pinn = zeros(1,Npoints);
for k=1:Npoints
	s = 1i*omega(k);
	A_val = polyval(A_poly_for_polyval,s);	
    B_val = polyval(B_poly_for_polyval,s);
    A_val_pinn = polyval(A_poly_for_polyval_pinn,s);
    B_val_pinn = polyval(B_poly_for_polyval_pinn,s);
	Z_jw(k) = A_val./B_val;
    Z_jw_pinn(k) = A_val_pinn./B_val_pinn;
end

fig = figure('Name','Impedance Frequency Response','Color','w');
subplot(2,1,1)
plot(f, abs(Z_jw),  'LineWidth',1.5, 'DisplayName', 'True')
hold on
plot(f, abs(Z_jw_pinn), '--','LineWidth',1.5, 'DisplayName', 'Recovered','MarkerIndices', 1:2:length(Z_jw_pinn))
xlabel('Frequency (Hz)')
ylabel('|Z(j\omega)|')
grid on
legend('Location','best')
subplot(2,1,2)
plot(f, angle(Z_jw)*180/pi,'LineWidth',1.5, 'DisplayName', 'True');
hold on
plot(f, angle(Z_jw_pinn)*180/pi,'--','LineWidth',1.5, 'DisplayName', 'Recovered','MarkerIndices', 1:2:length(Z_jw_pinn));
xlabel('Frequency(Hz)')
ylabel('Phase of Z(jw)')		
grid on
legend('Location','best')

exportgraphics(fig, 'ImpedanceFrequencyResponse.jpg');