clc; clear; close all;

% ==========================================
% 1. CONFIGURATION
% ==========================================
AT_BIN = 'D:\atWin10_2020_11_4\atWin10_2020_11_4\windows-bin-20201102\';
filename = 'mysim';

% ==========================================
% 2. WRITE ENV & RUN BELLHOP
% ==========================================
if exist([filename '.shd'], 'file'), delete([filename '.shd']); end
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

exe_path = fullfile(AT_BIN, 'bellhop.exe');
[status, cmdout] = system(sprintf('"%s" %s', exe_path, filename));
if status ~= 0, error('Bellhop Failed: %s', cmdout); end

% ==========================================
% 3. THE "VARIANCE PLATEAU" DECODER
% ==========================================
fid = fopen([filename '.shd'], 'rb');
raw = fread(fid, inf, 'float32');
fclose(fid);

Nrr = 201; 
Nrd = 101; 
Stride = length(raw) / Nrr;

% 1. Reshape to Raw Matrix (Includes Data + Padding)
RawMat = reshape(raw, Stride, Nrr);

% 2. Calculate Variance of each ROW (Ignoring the "Silence" zone at start)
% We check columns 50 to end to avoid the initial "Red Zone" biasing the check
CheckRegion = RawMat(:, 50:end);
RowVar = var(CheckRegion, 0, 2);

% 3. Find the longest continuous block of "Active" rows
% Active = Variance > small threshold (not constant/zero) AND < huge threshold (not garbage integer)
IsActive = (RowVar > 1e-12) & (RowVar < 1e5);

% Scan for longest sequence
current_len = 0; max_len = 0; start_idx = 1; best_start = 1;
for i = 1:length(IsActive)
    if IsActive(i)
        if current_len == 0, start_idx = i; end
        current_len = current_len + 1;
    else
        if current_len > max_len
            max_len = current_len;
            best_start = start_idx;
        end
        current_len = 0;
    end
end
% Check end of loop
if current_len > max_len
    max_len = current_len;
    best_start = start_idx;
end

fprintf('[3] Detected Data Block: Rows %d to %d (Length: %d)\n', ...
    best_start, best_start + max_len - 1, max_len);

% 4. Sanity Check & Extract
% We expect roughly 202 rows (101 Real + 101 Imag).
% Sometimes Bellhop adds 2 padding floats, making the "active" block slightly larger or smaller.
% We force extract exactly 202 rows from the center of the detected block.

CenterOfBlock = best_start + floor(max_len/2);
ExtractStart = CenterOfBlock - 101; % (202 / 2 = 101)
ExtractEnd = ExtractStart + 201;    % Total 202 rows

if ExtractStart < 1 || ExtractEnd > Stride
    warning('Block alignment fuzzy. Using Best Start directly.');
    ExtractStart = best_start;
    ExtractEnd = best_start + 201;
end

CleanData = RawMat(ExtractStart : ExtractEnd, :);

% ==========================================
% 4. PLOT
% ==========================================
p_complex = CleanData(1:2:end, :) + 1i * CleanData(2:2:end, :);

r_coords = linspace(1, 1000, Nrr);
z_coords = linspace(0, 100, Nrd);
TL = -20 * log10(abs(p_complex) + 1e-12);

figure;
imagesc(r_coords, z_coords, TL);
colormap(jet); colorbar;
set(gca, 'YDir', 'reverse');
clim([40 90]); % Correct Scaling
title('Transmission Loss (150 Hz)');
xlabel('Range (m)'); ylabel('Depth (m)');