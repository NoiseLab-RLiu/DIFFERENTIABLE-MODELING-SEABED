clc; clear; close all;

% ==========================================
% 1. SETUP
% ==========================================
AT_BIN = 'D:\atWin10_2020_11_4\atWin10_2020_11_4\windows-bin-20201102\';
filename = 'mysim';

% Re-run simulation to ensure fresh file
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
% 2. THE CORRECTED FILTER
% ==========================================
fid = fopen([filename '.shd'], 'rb');
raw = fread(fid, inf, 'float32');
fclose(fid);

Nrr = 201; 
Nrd = 101; 
Stride = length(raw) / Nrr; % 222

RawMat = reshape(raw, Stride, Nrr);

% Step A: Calculate Average Magnitude
% Garbage Rows (Indices) -> Mean > 10.0
% Data Rows (Pressure)   -> Mean < 10.0
RowMean = mean(abs(RawMat), 2);

% Step B: Apply Filter (Small Magnitude AND Non-Empty)
is_valid_data = (RowMean < 10.0) & (RowMean > 1e-9);

% Step C: Get Indices
keep_idx = find(is_valid_data);

fprintf('Original Rows: %d.  Kept Rows: %d (Target: 202)\n', Stride, length(keep_idx));

% Step D: Safety Check (Force exactly 202 rows if filter is slightly off)
if length(keep_idx) ~= (Nrd*2)
    warning('Filter count mismatch. Forcing center selection.');
    % If we have > 202, the "real" data is usually in the middle of the block
    if length(keep_idx) > 202
        mid_point = floor(length(keep_idx)/2);
        start_cut = mid_point - 100; 
        if start_cut < 1, start_cut = 1; end
        keep_idx = keep_idx(start_cut : start_cut + 201);
    else
        % Fallback: If filter failed, manually skip the first 2 rows (common header)
        % and take the next 202.
        keep_idx = 3:(2+Nrd*2); 
    end
end

% Step E: CRITICAL - Restore Order
% We sort the INDICES to ensure we grab the rows from top to bottom
% as they appeared in the file.
keep_idx = sort(keep_idx, 'ascend');

% Step F: Extract
CleanData = RawMat(keep_idx, :);

% ==========================================
% 3. PLOT
% ==========================================
p_complex = CleanData(1:2:end, :) + 1i * CleanData(2:2:end, :);

r_coords = linspace(1, 1000, Nrr);
z_coords = linspace(0, 100, Nrd);
TL = -20 * log10(abs(p_complex) + 1e-12);

figure;
imagesc(r_coords, z_coords, TL);
colormap(jet); 
colorbar;
set(gca, 'YDir', 'reverse');
clim([40 90]); 
title('Transmission Loss');
xlabel('Range (m)'); ylabel('Depth (m)');

%%
clc; clear; close all;

% ==========================================
% 1. LOAD RAW DATA
% ==========================================
filename = 'mysim';
fid = fopen([filename '.shd'], 'rb');
if fid < 0, error('File not found. Run simulation first.'); end
raw = fread(fid, inf, 'float32');
fclose(fid);

Nrr = 201; 
Nrd = 101; 
Stride = length(raw) / Nrr; % 222

% Reshape to Raw Matrix
RawMat = reshape(raw, Stride, Nrr);

% ==========================================
% 2. EXTRACT THE CONTIGUOUS BLOCK
% ==========================================
% As you found, the valid rows are a chunk in the middle (e.g. 12 to 215).
% We use the Magnitude Filter to find the start of this chunk.
RowMean = mean(abs(RawMat), 2);
is_data = (RowMean < 10.0) & (RowMean > 1e-9);
valid_indices = find(is_data);

if isempty(valid_indices)
    error('Filter failed. No data found.');
end

% Use the FIRST valid index as the anchor point.
% We need exactly 202 rows (101 Real + 101 Imag).
StartIdx = valid_indices(1);
EndIdx = StartIdx + 201;

fprintf('Extracting Block: Rows %d to %d\n', StartIdx, EndIdx);
Block = RawMat(StartIdx : EndIdx, :);

% ==========================================
% 3. GENERATE 4 PERMUTATIONS
% ==========================================
% We will test: 
% 1. Standard alignment
% 2. Shifted alignment (Fixes "Off-by-one" Real/Imag swap)
% 3. Transposed Standard
% 4. Transposed Shifted

r_coords = linspace(1, 1000, Nrr);
z_coords = linspace(0, 100, Nrd);

figure('Position', [100 100 1000 800], 'Name', 'Matrix Solver');

% --- CASE 1: STANDARD ---
p = Block(1:2:end, :) + 1i * Block(2:2:end, :);
TL = -20*log10(abs(p) + 1e-12);

subplot(2,2,1);
imagesc(r_coords, z_coords, TL);
colormap(jet); clim([40 90]);
title('1. Standard');
set(gca, 'YDir', 'reverse');

% --- CASE 2: SHIFTED START ---
% Try starting 1 row later (swaps Real/Imag pairing)
BlockShift = RawMat(StartIdx+1 : EndIdx+1, :);
p = BlockShift(1:2:end, :) + 1i * BlockShift(2:2:end, :);
TL = -20*log10(abs(p) + 1e-12);

subplot(2,2,2);
imagesc(r_coords, z_coords, TL);
colormap(jet); clim([40 90]);
title('2. Shifted (+1 Offset)');
set(gca, 'YDir', 'reverse');

% --- CASE 3: TRANSPOSED ---
% Maybe Bellhop wrote [Range x Depth] instead of [Depth x Range]
% We assume the Standard block, but flip X and Y.
p = Block(1:2:end, :) + 1i * Block(2:2:end, :);
TL = -20*log10(abs(p) + 1e-12);

subplot(2,2,3);
% Note: We swap r_coords and z_coords axes here
imagesc(z_coords, r_coords, TL'); 
colormap(jet); clim([40 90]);
title('3. Transposed');
set(gca, 'YDir', 'normal'); % Range is usually X, so normal YDir often fits transpose better

% --- CASE 4: TRANSPOSED + SHIFTED ---
BlockShift = RawMat(StartIdx+1 : EndIdx+1, :);
p = BlockShift(1:2:end, :) + 1i * BlockShift(2:2:end, :);
TL = -20*log10(abs(p) + 1e-12);

subplot(2,2,4);
imagesc(z_coords, r_coords, TL'); 
colormap(jet); clim([40 90]);
title('4. Transposed + Shifted');
set(gca, 'YDir', 'normal');

sgtitle('Which one looks like a Beam?');
%%
clc; clear; close all;

% ==========================================
% 1. LOAD RAW DATA
% ==========================================
filename = 'mysim';
if ~exist([filename '.shd'], 'file'), error('SHD file missing.'); end

fid = fopen([filename '.shd'], 'rb');
raw = fread(fid, inf, 'float32');
fclose(fid);

Nrr = 201; 
Stride = 222; % We have confirmed this is correct

% Reshape to Matrix [222 Rows x 201 Columns]
RawMat = reshape(raw, Stride, Nrr);

% ==========================================
% 2. THE VISUAL BIOPSY
% ==========================================
% We calculate the Average Magnitude of every row index (1 to 222)
RowMean = mean(abs(RawMat), 2);

figure('Name', 'File Structure Biopsy', 'Color', 'w');

% PLOT 1: The Full Scale (To see Indices/Huge Garbage)
subplot(2, 1, 1);
plot(1:Stride, RowMean, 'b.-', 'LineWidth', 1);
grid on;
title('File Structure: Row Average (Full Scale)');
xlabel('Row Index'); ylabel('Average Value');
xlim([1 222]);

% PLOT 2: The "Physics" Scale (Zoomed in)
% Physics data (Pressure) is usually between 0.0 and 1.0
subplot(2, 1, 2);
plot(1:Stride, RowMean, 'r.-', 'LineWidth', 1);
grid on;
title('Zoomed In: Look for the "Valley" (Values < 2.0)');
xlabel('Row Index'); ylabel('Average Value');
xlim([1 222]);
ylim([0 2.0]); % Zoom to see only physics data

% Mark the "Data" rows with green circles
hold on;
DataRows = find(RowMean > 1e-9 & RowMean < 2.0);
plot(DataRows, RowMean(DataRows), 'go');
legend('Row Mean', 'Probable Data');

fprintf('--- AUTOMATIC DETECTION ---\n');
fprintf('Detected %d rows that look like physics data (< 2.0).\n', length(DataRows));
fprintf('Indices: %d to %d\n', min(DataRows), max(DataRows));
fprintf('---------------------------\n');

%%
clc; clear; close all;

% ==========================================
% 1. LOAD RAW DATA
% ==========================================
filename = 'mysim';
if ~exist([filename '.shd'], 'file'), error('SHD file missing.'); end

fid = fopen([filename '.shd'], 'rb');
raw = fread(fid, inf, 'float32');
fclose(fid);

Nrr = 201; 
Stride = 222; 
RawMat = reshape(raw, Stride, Nrr);

% ==========================================
% 2. THE "ACTIVITY" FILTER
% ==========================================
% Headers and Depth Vectors are identical in every column.
% Physics Data CHANGES from column to column (Range to Range).
% We calculate the Standard Deviation across the ranges.
RowActivity = std(RawMat, 0, 2);

% A. Plot the Activity Profile (Diagnostic)
figure;
subplot(2,1,1);
plot(RowActivity, 'k.-');
title('Row Activity (Standard Deviation)');
xlabel('Row Index'); ylabel('Variation');
grid on;

% B. Select only "Active" Rows
% - Garbage/Headers have variation == 0.
% - Data has variation > 0.
% We use a small threshold to catch the real signals.
is_active = RowActivity > 1e-6; 
keep_idx = find(is_active);

fprintf('Found %d active rows.\n', length(keep_idx));

% C. Handle Mismatches
% We expect exactly 202 rows (101 Real + 101 Imag).
% If we found more/less, we need to be smart.
if length(keep_idx) ~= 202
    fprintf('Warning: Expected 202 rows, found %d.\n', length(keep_idx));
    
    % If we found exactly 101, maybe the data is not Complex?
    if length(keep_idx) == 101
        fprintf('Assuming data is Magnitude-Only (Real part).\n');
        p_complex = RawMat(keep_idx, :); % Treat as real pressure
    
    % If we have a lot (e.g. 210), crop the edges (headers often have slight jitter)
    elseif length(keep_idx) > 202
        center = round(mean(keep_idx));
        start_cut = center - 100;
        keep_idx = start_cut : (start_cut + 201);
        p_real = RawMat(keep_idx(1:2:end), :);
        p_imag = RawMat(keep_idx(2:2:end), :);
        p_complex = p_real + 1i * p_imag;
    else
        % Fallback: Just try to use whatever we found
        % This effectively deletes the empty rows and squashes the rest together
        CleanData = RawMat(keep_idx, :);
        % Attempt standard de-interleaving
        try
            p_complex = CleanData(1:2:end, :) + 1i * CleanData(2:2:end, :);
        catch
            error('Odd number of active rows. Cannot pair Real/Imag.');
        end
    end
else
    % Perfect match (202 rows)
    fprintf('Perfect match found! Extracting physics data.\n');
    CleanData = RawMat(keep_idx, :);
    p_complex = CleanData(1:2:end, :) + 1i * CleanData(2:2:end, :);
end

% ==========================================
% 3. PLOT
% ==========================================
% Ensure we have dimensions to match the data we extracted
[FinalNrd, ~] = size(p_complex);
r_coords = linspace(1, 1000, Nrr);
z_coords = linspace(0, 100, FinalNrd); 

TL = -20 * log10(abs(p_complex) + 1e-12);

subplot(2,1,2);
imagesc(r_coords, z_coords, TL);
colormap(jet); 
colorbar;
set(gca, 'YDir', 'reverse');
clim([40 90]); 
title(sprintf('Reconstructed Beam (%d Depths)', FinalNrd));
xlabel('Range (m)'); ylabel('Depth (m)');