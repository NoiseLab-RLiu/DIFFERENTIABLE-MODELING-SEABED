clc;
filename = 'pekeris_kraken';
AT_BIN = 'D:\atWin10_2020_11_4\atWin10_2020_11_4\windows-bin-20201102\';

% 1. CHECK IF MODES EXIST
mod_file = [filename '.mod'];
if exist(mod_file, 'file')
    d = dir(mod_file);
    fprintf('MOD File Size: %d bytes\n', d.bytes);
    if d.bytes < 100
        fprintf('WARNING: MOD file is too small. KRAKEN likely found 0 modes.\n');
    end
else
    fprintf('ERROR: MOD file does not exist. KRAKEN failed completely.\n');
end

% 2. RUN FIELD MANUALLY WITH VERBOSE OUTPUT
fprintf('\n--- RERUNNING FIELD WITH OUTPUT ---\n');
field_exe = fullfile(AT_BIN, 'field.exe');
cmd_f = sprintf('"%s" %s', field_exe, filename);
[status, output] = system(cmd_f);
disp(output); % <--- THIS WILL SHOW THE REAL ERROR

% 3. CHECK SHD FILE
shd_file = [filename '.shd'];
if exist(shd_file, 'file')
    fprintf('SUCCESS: SHD file created (%s)\n', shd_file);
    % Try reading without extension
    try
        read_shd(filename);
        fprintf('read_shd successful!\n');
    catch ME
        fprintf('read_shd failed: %s\n', ME.message);
    end
else
    fprintf('FAILURE: Still no SHD file.\n');
end