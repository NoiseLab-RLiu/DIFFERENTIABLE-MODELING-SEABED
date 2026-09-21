clc; clear; %close all;

% =========================================================================
% STANDARD PEKERIS WAVEGUIDE EXAMPLE (OALIB)
% Sources: OALIB / NPS Lab 4
% =========================================================================

% >>> 1. SETUP: Point to your BELLHOP.EXE <<<
% Update this path to where your bellhop.exe is located
AT_BIN = 'D:\atWin10_2020_11_4\atWin10_2020_11_4\windows-bin-20201102\';
filename = 'pekeris_sim';

% =========================================================================
% 2. GENERATE ENV FILE
% Standard OALIB Parameters:
% - Water Depth: 100m, Sound Speed: 1500 m/s (Isovelocity)
% - Bottom: 1600 m/s, Density: 1.8, Attenuation: 0.5
% =========================================================================
fid = fopen([filename '.env'], 'w');
fprintf(fid, '''Pekeris Waveguide''\n');
fprintf(fid, '150.0\n');        % Frequency: 150 Hz
fprintf(fid, '1\n');            % NMedia
fprintf(fid, '''CVW''\n');      % Opt: Coherent Pressure, Vac Surface, Wavelength Atten
fprintf(fid, '51 0.0 100.0\n'); % SSP Interpolation: 51 pts
fprintf(fid, '0.0    1500.0 0.0 1.0 0.0 0.0\n'); % Water Top
fprintf(fid, '100.0  1500.0 0.0 1.0 0.0 0.0\n'); % Water Bottom
fprintf(fid, '''A'' 0.0\n');    % Bottom: Acousto-Elastic
%fprintf(fid, '100.0  2000.0 0.0 1.8 0.5 0.0\n'); % Sediment Params
%fprintf(fid, '100.0  1600.0 0.0 1.8 0.5 0.0\n'); % Sediment Params
fprintf(fid, '100.0  2000.0 0.0 1.8 0.0 0.0\n'); % Sediment Params
%fprintf(fid, '100.0  2000.0 0.0 10.8 0.0 0.0\n'); % Sediment Params
fprintf(fid, '1\n');            % NSources
fprintf(fid, '25.0\n');         % Source Depth
fprintf(fid, '101\n');          % NReceivers (Depths)
fprintf(fid, '0.0 100.0 /\n');  % Receiver Depth Grid
fprintf(fid, '1001\n');          % NReceivers (Ranges)
fprintf(fid, '0.001 5.0 /\n');  % Receiver Range Grid (1m to 1km)
fprintf(fid, '''CB''\n');       % Run Type: Coherent Beams
fprintf(fid, '0\n');            % NBeams (0 = Auto)
fprintf(fid, '-80.0 80.0 /\n'); % Beam Angles
fprintf(fid, '0.0 101.0 5.0\n');% Box: Auto step, 101m depth, 1km range
fclose(fid);
fprintf('[1] ENV file generated: %s.env\n', filename);

% =========================================================================
% 3. RUN BELLHOP
% =========================================================================
% Ensure previous output is gone
if exist([filename '.shd'], 'file'), delete([filename '.shd']); end

exe_path = fullfile(AT_BIN, 'bellhop.exe');
%exe_path = fullfile(AT_BIN, 'kraken.exe');
cmd = sprintf('"%s" %s', exe_path, filename);
[status, cmdout] = system(cmd);

if status ~= 0
    error('Bellhop Execution Failed: %s', cmdout);
end
fprintf('[2] Bellhop run successful.\n');

% =========================================================================
% 4. PLOT (Requires read_shd.m)
% =========================================================================
if ~exist('read_shd', 'file')
    error('MISSING FILE: You must download read_shd.m and place it in this folder.');
end

% FIX: Add the '.shd' extension explicitly
[PlotTitle, PlotType, Freq, Atten, Pos, p] = read_shd([filename '.shd']);
p_2d = squeeze(p);
% Extract Grid
r_coords = Pos.r.range;
z_coords = Pos.r.depth;

% Calculate Transmission Loss (TL)
% Formula: TL = -20 * log10( |Pressure| )
% We add a small epsilon (1e-20) to avoid log(0) errors
TL = -20 * log10(abs(p_2d) + 1e-20);

% Visualization
figure('Name', 'OALIB Pekeris Example');
imagesc(r_coords, z_coords, TL);
colormap(jet);
colorbar;
set(gca, 'YDir', 'reverse');

% Standard range for shallow water (Red=Quiet, Blue=Loud)
clim([40 80]); 

title(PlotTitle);
xlabel('Range (m)');
ylabel('Depth (m)');