clc; clear;
% =========================================================================
% LAYERED KRAKEN: THE ACTUAL WORKING FIELD.EXE PIPELINE
% =========================================================================

fclose('all');
warning('off', 'all'); 
delete('K_*.env'); delete('K_*.prt'); delete('K_*.mod'); 
delete('K_*.flp'); delete('K_*.shd'); 
warning('on', 'all');

AT_ROOT = 'D:\atWin10_2020_11_4\atWin10_2020_11_4\';
AT_BIN  = fullfile(AT_ROOT, 'windows-bin-20201102');
addpath(genpath(fullfile(AT_ROOT, 'Matlab'))); 

FREQ_LIST = [30.0, 35.0, 50.0, 100.0, 150.0, 200, 250];

D     = 100.0;  
cb    = 2026.0;       
rhob  = 1.3;        
alphab= 0.8;        
zs    = 25.0;   

Nz = 101;      z_min = 0.0;   z_max = 100.0;
Nr = 1001;     r_min = 0.001; r_max = 5.0; % FIELD requires range in km!

for f_idx = 1:length(FREQ_LIST)
    freq = FREQ_LIST(f_idx);
    
    fprintf('\n>>> PROCESSING FREQUENCY: %0.1f Hz <<<\n', freq);
    filename = sprintf('K_%d', round(freq)); 
    
    % ---------------------------------------------------------------------
    % A. GENERATE ENV FILE 
    % ---------------------------------------------------------------------
    fid = fopen([filename '.env'], 'w');
    fprintf(fid, '''Layered MultiFreq''\n');
    fprintf(fid, '%f\n', freq);
    fprintf(fid, '1\n');            
    fprintf(fid, '''CVW''\n');   
    
    fprintf(fid, '2001 0.0 %0.3f\n', D); 
    fprintf(fid, '  0.0  1520.0 0.0 1.0 0.0 0.0\n');  
    fprintf(fid, ' 33.0  1510.0 0.0 1.0 0.0 0.0\n');  
    fprintf(fid, ' 67.0  1490.0 0.0 1.0 0.0 0.0\n');  
    fprintf(fid, '100.0  1480.0 0.0 1.0 0.0 0.0\n');  
    
    fprintf(fid, '''A'' 0.0\n');    
    fprintf(fid, '%0.3f  %0.3f 0.0 %0.3f %0.3f 0.0\n', D, cb, rhob, alphab); 
    fprintf(fid, '1400.0  %0.1f\n', cb); % Strictly trapped modes
    fprintf(fid, '10.0\n'); 
    
    fprintf(fid, '1\n'); fprintf(fid, '%0.1f /\n', zs);
    fprintf(fid, '%d\n', Nz); 
    fprintf(fid, '%0.1f %0.1f /\n', z_min, z_max); 
    fclose(fid);
    
    % ---------------------------------------------------------------------
    % B. RUN KRAKEN & PARSE PRT FOR MODE COUNT
    % ---------------------------------------------------------------------
    kraken_exe = fullfile(AT_BIN, 'kraken.exe');
    system(sprintf('"%s" %s', kraken_exe, filename));
    
    fid = fopen([filename '.prt'], 'r');
    modes_found = false; num_modes = 0;
    while ~feof(fid)
        line = fgetl(fid);
        if contains(line, 'I    k (1/m)'), modes_found = true; continue; end
        if modes_found
            data = sscanf(line, '%d %f %f %f %f');
            if length(data) >= 3
                num_modes = num_modes + 1;
            elseif isempty(line) || contains(line, '___')
                if num_modes > 0, break; end 
            end
        end
    end
    fclose(fid);
    
    if num_modes == 0
        fprintf('[ERROR] KRAKEN found 0 trapped modes! Skipping.\n'); continue;
    end
    fprintf('    -> KRAKEN found %d trapped modes.\n', num_modes);

    % ---------------------------------------------------------------------
    % C. GENERATE FLP FILE (WITH EXPLICIT MODE LIST!)
    % ---------------------------------------------------------------------
    fid = fopen([filename '.flp'], 'w');
    fprintf(fid, '''Layered TL''\n');
    fprintf(fid, '''R''\n');                  
    fprintf(fid, '%d\n', num_modes);          
    fprintf(fid, '%d ', 1:num_modes); % THE CRITICAL MISSING LINE
    fprintf(fid, '/\n');                      
    fprintf(fid, '1\n');                      
    fprintf(fid, '%0.1f /\n', zs);            
    fprintf(fid, '%d\n', Nz);                 
    fprintf(fid, '%0.1f %0.1f /\n', z_min, z_max); 
    fprintf(fid, '%d\n', Nr);                 
    fprintf(fid, '%0.3f %0.3f /\n', r_min, r_max); 
    fclose(fid);

    % ---------------------------------------------------------------------
    % D. RUN FIELD & READ SHD
    % ---------------------------------------------------------------------
    field_exe = fullfile(AT_BIN, 'field.exe');
    system(sprintf('"%s" %s', field_exe, filename));
    
    shd_file = [filename '.shd'];
    if exist(shd_file, 'file')
        [~, ~, ~, ~, Pos, pressure] = read_shd(shd_file);
        
        P_2D = squeeze(pressure); 
        
        % The math correction for positive values
        TL_2D = -20 * log10(abs(P_2D) + 1e-20);
        
        z_vec = Pos.r.depth;
        r_vec = Pos.r.range * 1000; % Convert km to m
        
        out_name = sprintf('TL_kraken_layered_%d.mat', round(freq));
        save(out_name, 'TL_2D', 'P_2D', 'r_vec', 'z_vec', 'freq', 'cb', 'rhob', 'alphab');
        fprintf('    -> Saved %s\n', out_name);
        
        delete([filename '.*']);
    else
        fprintf('[ERROR] FIELD failed to create SHD file.\n');
        type([filename '.prt']);
    end
end
fprintf('\n[DONE] All frequencies processed.\n');