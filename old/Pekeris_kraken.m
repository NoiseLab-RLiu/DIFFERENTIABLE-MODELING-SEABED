clc; clear;

% =========================================================================
% PEKERIS KRAKEN (Fixed Path for read_shd_bin)
% =========================================================================

% >>> 1. SETUP PATHS <<<
AT_ROOT = 'D:\atWin10_2020_11_4\atWin10_2020_11_4\';
AT_BIN  = fullfile(AT_ROOT, 'windows-bin-20201102');

% FIX: Use genpath to add "Matlab" AND "Matlab\ReadWrite" to path
AT_MATLAB = fullfile(AT_ROOT, 'Matlab');
if exist(AT_MATLAB, 'dir')
    addpath(genpath(AT_MATLAB)); % <--- Adds all subfolders automatically
else
    error('Could not find "Matlab" folder at: %s', AT_MATLAB);
end

filename = 'pekeris_kraken';

% =========================================================================
% 2. GENERATE ENV FILE
% =========================================================================
fid = fopen([filename '.env'], 'w');

fprintf(fid, '''Pekeris KRAKEN''\n');
fprintf(fid, '150.0\n');        % Freq
fprintf(fid, '1\n');            % NMedia
fprintf(fid, '''CVW''\n');      % Opts

% Mesh: 2001 points
fprintf(fid, '2001 0.0 100.0\n'); 

% Water SSP
fprintf(fid, '0.000    1500.0 0.0 1.0 0.0 0.0\n'); 
fprintf(fid, '100.000  1500.0 0.0 1.0 0.0 0.0\n'); 

% Bottom Half-Space (Hardcoded Params: D=100, Cb=2000, Rho=1.8, Alpha=0.5)
fprintf(fid, '''A'' 0.0\n');    
fprintf(fid, '100.000  2000.0 0.0 1.8 0.5 0.0\n'); 

% Phase Speed Limits
fprintf(fid, '1400.0  3000.0\n'); 
% Max Range
fprintf(fid, '10.0\n'); 

% Sources
fprintf(fid, '1\n');            
fprintf(fid, '25.0\n');         

% Receivers
fprintf(fid, '1\n');            
fprintf(fid, '50.0 /\n');       
fprintf(fid, '201\n');          
fprintf(fid, '1.0 5.0 /\n');    

fclose(fid);
fprintf('[1] ENV file generated.\n');

% =========================================================================
% 3. GENERATE FLP FILE
% =========================================================================
fid = fopen([filename '.flp'], 'w');
fprintf(fid, '''Pekeris Field''\n');
fprintf(fid, '''C''\n');        
fprintf(fid, '1  9999\n');      
fprintf(fid, '1\n');            
fprintf(fid, '25.0\n');         
fprintf(fid, '1\n');            
fprintf(fid, '50.0 /\n');       
fprintf(fid, '201\n');          
fprintf(fid, '1.0 5.0 /\n');    
fclose(fid);
fprintf('[2] FLP file generated.\n');

% =========================================================================
% 4. RUN KRAKEN & FIELD
% =========================================================================
if exist([filename '.shd'], 'file'), delete([filename '.shd']); end
if exist([filename '.mod'], 'file'), delete([filename '.mod']); end

kraken_exe = fullfile(AT_BIN, 'kraken.exe');
field_exe  = fullfile(AT_BIN, 'field.exe');

% A. KRAKEN
cmd = sprintf('"%s" %s', kraken_exe, filename);
[status, cmdout] = system(cmd);
if status ~= 0, error('KRAKEN Failed: %s', cmdout); end
fprintf('[3] KRAKEN run successful.\n');

% B. FIELD
cmd = sprintf('"%s" %s', field_exe, filename);
[status, cmdout] = system(cmd);
if status ~= 0, error('FIELD Failed: %s', cmdout); end
fprintf('[4] FIELD run successful.\n');

% =========================================================================
% 5. SAVE DATA
% =========================================================================
% This will now work because 'ReadWrite' is in the path
[PlotTitle, PlotType, Freq, Atten, Pos, p] = read_shd([filename '.shd']);
p_vec = squeeze(p);
r_vec = Pos.r.range;

% Calculate TL
TL_kraken = -20 * log10(abs(p_vec) + 1e-20);

% Save
save('TL_data_kraken.mat', 'TL_kraken', 'r_vec');
fprintf('[5] SUCCESS! Data saved to TL_data_kraken.mat\n');

% Plot
figure; plot(r_vec, TL_kraken, 'b');
title('KRAKEN Ground Truth');
xlabel('Range (m)'); ylabel('TL (dB)');
set(gca, 'YDir', 'reverse'); grid on;