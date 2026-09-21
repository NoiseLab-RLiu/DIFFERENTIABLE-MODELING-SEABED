%%  Pekeris waveguide – Newton verification  (element-wise safe)
%clear; close all; clc;

% ─── constants ─────────────────────────────────────────────────────────
f      = 100;                       % Hz
omega  = 2*pi*f;                    % rad/s
c_w    = 1500;                      % m/s
c_b    = 2000;                      % m/s
D      = 100;                       % m

k0 = omega/c_w;
kb = omega/c_b;

% ─── helper anonymous functions (always element-wise) ─────────────────
gamma   = @(k) sqrt(omega^2/c_w^2 - k.^2);
gamma_b = @(k) sqrt(k.^2 - omega^2/c_b^2);

f_disp = @(k)  tan(gamma(k).*D) + gamma(k)./gamma_b(k);

df_dk  = @(k) (1./cos(gamma(k).*D)).^2 .* (-k./gamma(k)) .* D ...    % term-1
              - k ./ (gamma(k).*gamma_b(k)) ...                      % term-2
              - gamma(k).*k ./ (gamma_b(k).^3);                      % term-3

% ─── initial asymptotic guesses  k^(0)_m  (m = 0…8) ───────────────────
m        = 0:8;
k_guess  = sqrt((omega/c_w)^2 - ((m+0.52)*pi/D).^2);   % (1×9)

% storage for trajectories
maxIter  = 15;
k_hist   = nan(maxIter+1, numel(k_guess));
k_hist(1,:) = k_guess;

% ─── Newton loop (vectorised) ─────────────────────────────────────────
k = k_guess;
for it = 1:maxIter
    k = k - f_disp(k) ./ df_dk(k);     % element-wise update
    k_hist(it+1,:) = k;
end

% ─── plot convergence ────────────────────────────────────────────────
figure;  hold on; box on;
iters = 0:maxIter;
for j = 1:size(k_hist,2)
    plot(iters, k_hist(:,j), '-o', 'LineWidth',1.5, ...
         'DisplayName', sprintf('mode %d', j-1));
end
xlabel('Newton iteration'); ylabel('k_m  (m^{-1})');
title('Newton convergence of the nine guided modes');
legend('show'); grid on;

% ─── print final roots ────────────────────────────────────────────────
disp('Converged roots k_m (1/m):');
disp(k_hist(end,:).');
