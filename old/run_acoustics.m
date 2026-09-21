clc; clear; close all;

% ==========================================
% 1. CONFIGURATION
% ==========================================
AT_BIN = 'D:\atWin10_2020_11_4\atWin10_2020_11_4\windows-bin-20201102\';
filename = 'mysim';

% ==========================================
% 2. WRITE ENV (Standard Binary Mode)
% ==========================================
fid = fopen([filename '.env'], 'w');
fprintf(fid, '''Simple Pekeris Waveguide''\n');
fprintf(fid, '150.0\n');
fprintf(fid, '1\n');      
fprintf(fid, '''CVW''\n'); 
fprintf(fid, '51 0.0 100.0\n'); 
fprintf(fid, '0.0    1500.0 0.0 1.0 0.0 0.0\n');
fprintf(fid, '100.0  1500.0 0.0 1.0 0.0 0.0\n');
fprintf(fid, '''A'' 0.0\n'); 
fprintf(fid, '100.0  2000.0 0.5 1.8 0.0 0.0\n');
fprintf(fid, '1\n');   
fprintf(fid, '25.0\n'); 
fprintf(fid, '101\n'); 
fprintf(fid, '0.0 100.0 /\n'); 
fprintf(fid, '201\n'); 
fprintf(fid, '0.001 1.0 /\n'); 
fprintf(fid, '''CB''\n'); 
fprintf(fid, '3000\n');   
fprintf(fid, '-80.0 80.0 /\n'); 
fprintf(fid, '0.0 101.0 1.0\n'); 
fclose(fid);
fprintf('[1] ENV written.\n');

% ==========================================
% 3. RUN BELLHOP
% ==========================================
if exist([filename '.shd'], 'file'), delete([filename '.shd']); end
exe_path = fullfile(AT_BIN, 'bellhop.exe');
cmd = sprintf('"%s" %s', exe_path, filename);
[status, cmdout] = system(cmd);
if status ~= 0, error('Bellhop Failed: %s', cmdout); end
fprintf('[2] Bellhop finished.\n');

% ==========================================
% 4. THE "SIEVE" READER (Universal Fix)
% ==========================================
fprintf('[3] Decoding binary with Sieve Method...\n');
fid = fopen([filename '.shd'], 'rb');
raw = fread(fid, inf, 'float32');
fclose(fid);

% 1. Auto-Detect Dimensions
Nrr = 201; % We know this from input
Nrd = 101; % We know this from input
Stride = length(raw) / Nrr;

% Verification
if floor(Stride) ~= Stride
    error('File size mismatch. Size: %d, Ranges: %d. Ratio: %f', length(raw), Nrr, Stride);
end

fprintf('    Matrix Stride detected: %d floats per range.\n', Stride);

% 2. Reshape to Raw Matrix
Matrix = reshape(raw, Stride, Nrr);

% 3. Filter "Garbage" Rows
% Real pressure data usually has a magnitude between 0 and 100.
% Header integers often look like 0 or huge numbers when read as floats.
row_mag = mean(abs(Matrix), 2);

% Keep rows that look like physics data (Non-zero, but not astronomical)
% Epsilon 1e-9 avoids exact zeros (padding). 1e4 avoids header integers.
valid_rows = (row_mag > 1e-9) & (row_mag < 1e4);

% We expect exactly 202 valid rows (101 Real + 101 Imag)
if sum(valid_rows) ~= (Nrd * 2)
    warning('Sieve found %d valid rows, expected %d. Trying fallback sort...', sum(valid_rows), Nrd*2);
    % Fallback: Just take the rows with highest variance (most information)
    row_var = var(Matrix, 0, 2);
    [~, idx] = sort(row_var, 'descend');
    % Sort indices back to preserve depth order
    keep_idx = sort(idx(1:Nrd*2)); 
    CleanData = Matrix(keep_idx, :);
else
    CleanData = Matrix(valid_rows, :);
end

% 4. Reconstruct Complex Pressure
p_real = CleanData(1:2:end, :);
p_imag = CleanData(2:2:end, :);
p_complex = p_real + 1i * p_imag;

fprintf('    Data Extracted. Grid: %d x %d\n', size(p_complex));

% ==========================================
% 5. PLOT
% ==========================================
r_coords = linspace(1, 1000, Nrr);
z_coords = linspace(0, 100, Nrd);

% Transmission Loss
% Cap at 90dB so "silence" (400) doesn't ruin the plot
TL = -20 * log10(abs(p_complex) + 1e-12);

figure;
imagesc(r_coords, z_coords, TL);
colormap(jet); 
colorbar;
set(gca, 'YDir', 'reverse');
clim([40 90]); % Force color range to "Audible" sound levels

title('Transmission Loss (150 Hz)');
xlabel('Range (m)'); 
ylabel('Depth (m)');

fprintf('[SUCCESS] Plot Generated.\n');