clc; clear;

% =========================================================================
% PEKERIS KRAKEN: MANUAL 2D RECONSTRUCTION (WITH 4*PI FACTOR)
% =========================================================================

% >>> 1. SETUP PATHS <<<
AT_ROOT = 'D:\atWin10_2020_11_4\atWin10_2020_11_4\';
AT_BIN  = fullfile(AT_ROOT, 'windows-bin-20201102');
filename = 'pekeris_manual_2d';

% Physics Parameters
freq = 150.0;
omega = 2 * pi * freq;
D     = 100.0;  
cw    = 1500.0; 
cb    = 2026.0; 
rhob  = 1.55;
rhow  = 1.0;    
alphab= 0.3;    
zs    = 25.0;   

% Output Grid
z_vec = linspace(0, 100.0, 101);     
r_vec = linspace(1.0, 5000.0, 1001); 

% =========================================================================
% 2. GENERATE ENV FILE 
% =========================================================================
fid = fopen([filename '.env'], 'w');
fprintf(fid, '''Pekeris Manual 2D''\n');
fprintf(fid, '%f\n', freq);
fprintf(fid, '1\n');            
fprintf(fid, '''CVW''\n');      

fprintf(fid, '2001 0.0 %0.3f\n', D); 
fprintf(fid, '0.000    %0.3f 0.0 1.0 0.0 0.0\n', cw); 
fprintf(fid, '%0.3f  %0.3f 0.0 1.0 0.0 0.0\n', D, cw); 

fprintf(fid, '''A'' 0.0\n');    
fprintf(fid, '%0.3f  %0.3f 0.0 %0.3f %0.3f 0.0\n', D, cb, rhob, alphab); 

fprintf(fid, '1400.0  3000.0\n'); 
fprintf(fid, '10.0\n'); 
fprintf(fid, '1\n'); fprintf(fid, '25.0\n');
fprintf(fid, '1\n'); fprintf(fid, '50.0 /\n');
fprintf(fid, '201\n'); fprintf(fid, '1.0 5.0 /\n');

fclose(fid);
fprintf('[1] ENV file generated.\n');

% =========================================================================
% 3. RUN KRAKEN ONLY
% =========================================================================
if exist([filename '.prt'], 'file'), delete([filename '.prt']); end

kraken_exe = fullfile(AT_BIN, 'kraken.exe');
cmd = sprintf('"%s" %s', kraken_exe, filename);
[status, cmdout] = system(cmd);

if status ~= 0, error('KRAKEN Failed: %s', cmdout); end
fprintf('[2] KRAKEN run successful.\n');

% =========================================================================
% 4. PARSE PRT FILE
% =========================================================================
fid = fopen([filename '.prt'], 'r');
modes_found = false;
k_list = [];
alpha_list = [];

while ~feof(fid)
    line = fgetl(fid);
    if contains(line, 'I    k (1/m)'), modes_found = true; continue; end
    
    if modes_found
        data = sscanf(line, '%d %f %f %f %f');
        if length(data) >= 3
            k_list(end+1) = data(2);
            alpha_list(end+1) = data(3);
        elseif isempty(line) || contains(line, '___'), break; end
    end
end
fclose(fid);

% Ensure positive attenuation
k_kraken = k_list + 1i * abs(alpha_list); 

fprintf('[3] Extracted %d modes.\n', length(k_kraken));

% =========================================================================
% 5. BUILD 2D PRESSURE FIELD (WITH 4*PI CORRECTION)
% =========================================================================
fprintf('[4] Computing 2D Field...\n');

k0 = omega / cw;
P_2D = zeros(length(z_vec), length(r_vec)); 

for m = 1:length(k_kraken)
    kr = k_kraken(m);
    
    kz = sqrt(k0^2 - kr^2);
    
    alpha_nep = alphab * freq / cb / 8.686;
    kb_c = (omega/cb) + 1i*alpha_nep;
    kzb = sqrt(kb_c^2 - kr^2);
    
    % Normalization
    term1 = D/2;
    term2 = sin(2*kz*D)/(4*kz);
    term3 = (rhow/rhob) * (sin(kz*D)^2) / (2*1i*kzb);
    ModeAmp = 1.0 / sqrt(term1 - term2 + term3);
    
    % Shapes
    Psi_z = ModeAmp * sin(kz * z_vec);  
    Psi_s = ModeAmp * sin(kz * zs);     
    
    % Propagation
    phase = kr * r_vec;
    Hankel_r = sqrt(2 ./ (pi * kr * r_vec)) .* exp(1i * (phase - pi/4)); 
    
    contribution = (1i / (4*rhow)) * Psi_s * (Psi_z.' * Hankel_r);
    P_2D = P_2D + contribution;
end

% [CRITICAL FIX] Multiply by 4*pi for correct Point Source Reference
TL_2D = -20 * log10(abs(P_2D) * 4 * pi + 1e-20);

% =========================================================================
% 6. SAVE AND PLOT
% =========================================================================
save('TL_data_kraken2.mat', 'TL_2D', 'r_vec', 'z_vec');
fprintf('[5] SUCCESS! Saved to TL_data_kraken.mat\n');

figure('Name', 'Kraken 2D Final');
imagesc(r_vec, z_vec, TL_2D);
colormap(jet); 
h = colorbar; ylabel(h, 'TL (dB)');
title(['KRAKEN 2D TL (Corrected 4\pi Normalization)']);
xlabel('Range (m)'); ylabel('Depth (m)');

% Reset Color Scale to match Bellhop Standard
clim([40 80]);