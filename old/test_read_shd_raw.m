clc; close all;
filename = 'mysim.shd';

% --- 1. CONFIGURATION ---
Nrd = 101; % We know this from your setup
Nrr = 201; % We know this from your setup

% --- 2. READ RAW DATA ---
fid = fopen(filename, 'rb');
if fid == -1, error('File not found'); end

% Read entire file as float32
raw_floats = fread(fid, inf, 'float32');
fclose(fid);

fprintf('Total Floats Read: %d\n', length(raw_floats));

% --- 3. RESHAPE (The Magic Step) ---
% We calculated the stride is 888 bytes = 222 floats.
% If the file is perfect, length should be 222 * 201 = 44622.

Stride_Floats = 222;

if mod(length(raw_floats), Nrr) ~= 0
    warning('File length is not a clean multiple of 201 ranges. Trimming...');
    % Trim to nearest stride
    limit = floor(length(raw_floats) / Stride_Floats) * Stride_Floats;
    raw_floats = raw_floats(1:limit);
end

% Reshape into [Bytes_per_Column, Ranges]
% This aligns every column side-by-side.
RawGrid = reshape(raw_floats, Stride_Floats, Nrr);

% --- 4. VISUALIZE RAW GRID (DEBUG) ---
% This maps the file memory directly. You should see the wave pattern
% inside a specific block of rows, and "garbage/static" in other rows.
figure;
imagesc(abs(RawGrid));
title('Raw Memory Map (Find the Signal)');
xlabel('Range Step');
ylabel('Float Index (0-222)');
colorbar;

% --- 5. AUTOMATIC EXTRACTION ---
% We need 202 floats (101 Real + 101 Imag).
% The raw grid has 222 floats. 
% We find the block of 202 rows with the least NaNs and most signal.

% Calculate signal strength (variance) per row
row_variance = var(RawGrid, 0, 2, 'omitnan');

% The padding usually has 0 variance (constant markers).
% The data has high variance.
% We take the indices with the highest variance.
[~, sort_idx] = sort(row_variance, 'descend');

% Keep the top 202 active rows, then SORT them back to original order
% to preserve Real/Imag sequence.
keep_indices = sort(sort_idx(1:2*Nrd));

fprintf('Detected Data Rows: %d to %d (roughly)\n', min(keep_indices), max(keep_indices));

CleanGrid = RawGrid(keep_indices, :);

% --- 6. CONVERT TO COMPLEX ---
% Rows 1,3,5... are Real. Rows 2,4,6... are Imag.
p_complex = zeros(Nrd, Nrr);

real_part = CleanGrid(1:2:end, :);
imag_part = CleanGrid(2:2:end, :);

p_complex = real_part + 1i * imag_part;

% --- 7. CHECK & SAVE ---
max_p = max(max(abs(p_complex)));
fprintf('Max Pressure: %e\n', max_p);

if max_p == 0 || isnan(max_p)
    fprintf('WARNING: Data seems invalid. Check the Raw Memory Map figure.\n');
else
    % Plot Transmission Loss
    tl = -20*log10(abs(p_complex) + 1e-12);
    
    figure;
    z = linspace(0, 100, Nrd);
    r = linspace(0, 1000, Nrr);
    imagesc(r, z, tl);
    set(gca, 'YDir', 'reverse');
    colormap(jet); colorbar;
    caxis([40 100]);
    title('Recovered Transmission Loss');
    xlabel('Range (m)'); ylabel('Depth (m)');
    
    % EXPORT FOR PYTHON
    z_coords = z; r_coords = r;
    save('bellhop_data.mat', 'p_complex', 'z_coords', 'r_coords');
    fprintf('SUCCESS. Data saved to bellhop_data.mat\n');
end