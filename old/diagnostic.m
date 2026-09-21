clc; clear; close all;

% ==========================================
% 1. LOAD RAW DATA
% ==========================================
filename = 'mysim'; % Ensure this matches your file
if ~exist([filename '.shd'], 'file')
    error('SHD file not found. Run the simulation first.');
end

fid = fopen([filename '.shd'], 'rb');
raw = fread(fid, inf, 'float32');
fclose(fid);

% Known Dimensions
Nrr = 201; 
Nrd = 101;
Stride = length(raw) / Nrr; % Should be 222
fprintf('Stride: %.2f floats per range.\n', Stride);

% Reshape to Raw Matrix (222 rows x 201 cols)
RawMat = reshape(raw, Stride, Nrr);

% ==========================================
% 2. GENERATE ALIGNMENT "CONTACT SHEET"
% ==========================================
% The data block (202 rows) is floating somewhere inside the 222 rows.
% We will try 9 different starting offsets to find the right one.
offsets = [1, 3, 5, 7, 9, 11, 13, 15, 17];

figure('Name', 'Alignment Diagnostic', 'Color', 'w');

for k = 1:9
    start_row = offsets(k);
    
    % Extract 202 rows starting at this offset
    try
        block = RawMat(start_row : start_row + 201, :);
        
        % Form Complex Pressure
        p = block(1:2:end, :) + 1i * block(2:2:end, :);
        
        % Calculate Transmission Loss
        TL = -20 * log10(abs(p) + 1e-12);
        
        % Plot
        subplot(3, 3, k);
        imagesc(TL);
        colormap(jet); 
        set(gca, 'YDir', 'reverse', 'XTick', [], 'YTick', []);
        clim([40 90]); % Correct color limits
        title(sprintf('Offset +%d', start_row));
        
    catch
        subplot(3, 3, k);
        text(0.5, 0.5, 'Out of Bounds', 'HorizontalAlignment', 'center');
    end
end

sgtitle('Look for the smooth Beam Pattern');