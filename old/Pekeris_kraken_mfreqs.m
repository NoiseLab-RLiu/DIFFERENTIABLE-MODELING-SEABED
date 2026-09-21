clc; clear;

% =========================================================================
% PEKERIS KRAKEN: MULTI-FREQUENCY DATA GENERATOR
% =========================================================================

% >>> 1. SETUP <<<
AT_ROOT = 'D:\atWin10_2020_11_4\atWin10_2020_11_4\';
AT_BIN  = fullfile(AT_ROOT, 'windows-bin-20201102');

% Frequency List for Inversion
FREQ_LIST = [30.0, 35.0, 50.0, 100.0, 150.0, 200, 250];

% Physics Parameters (Dataset 2 Target)
D     = 100.0;  
cw    = 1500.0; 
cb    = 2026;%2000;%1910.0;   % Target CB
rhob  = 1.3;%1.8;%1.65;     % Target Rho
rhow  = 1.0;    
alphab= 0.8;%0.5;%0.45;      % Target Alpha
zs    = 25.0;   

% Common Grid (Must match for all freqs)
z_vec = linspace(0, 100.0, 101);     
r_vec = linspace(1.0, 5000.0, 1001); 

% =========================================================================
% 2. FREQUENCY LOOP
% =========================================================================
for f_idx = 1:length(FREQ_LIST)
    freq = FREQ_LIST(f_idx);
    omega = 2 * pi * freq;
    
    fprintf('\n>>> PROCESSING FREQUENCY: %0.1f Hz <<<\n', freq);
    
    % file naming: 'pekeris_30', 'pekeris_50' etc.
    filename = sprintf('pekeris_%d', round(freq));
    
    % ---------------------------------------------------------------------
    % A. GENERATE ENV FILE
    % ---------------------------------------------------------------------
    fid = fopen([filename '.env'], 'w');
    fprintf(fid, '''Pekeris MultiFreq''\n');
    fprintf(fid, '%f\n', freq);
    fprintf(fid, '1\n');            
    fprintf(fid, '''CVW''\n');      
    fprintf(fid, '2001 0.0 %0.3f\n', D); 
    fprintf(fid, '0.000    %0.3f 0.0 1.0 0.0 0.0\n', cw); 
    fprintf(fid, '%0.3f  %0.3f 0.0 1.0 0.0 0.0\n', D, cw); 
    fprintf(fid, '''A'' 0.0\n');    
    fprintf(fid, '%0.3f  %0.3f 0.0 %0.3f %0.3f 0.0\n', D, cb, rhob, alphab); 
    % Phase speed limits: widen range for lower freqs just in case
    fprintf(fid, '1400.0  10000.0\n'); 
    fprintf(fid, '10.0\n'); 
    fprintf(fid, '1\n'); fprintf(fid, '25.0\n');
    fprintf(fid, '1\n'); fprintf(fid, '50.0 /\n');
    fprintf(fid, '201\n'); fprintf(fid, '1.0 5.0 /\n');
    fclose(fid);
    
    % ---------------------------------------------------------------------
    % B. RUN KRAKEN
    % ---------------------------------------------------------------------
    if exist([filename '.prt'], 'file'), delete([filename '.prt']); end
    if exist([filename '.mod'], 'file'), delete([filename '.mod']); end
    
    kraken_exe = fullfile(AT_BIN, 'kraken.exe');
    cmd = sprintf('"%s" %s', kraken_exe, filename);
    [status, cmdout] = system(cmd);
    
    if status ~= 0
        fprintf('[ERROR] KRAKEN failed for %f Hz. Output:\n%s\n', freq, cmdout);
        continue; 
    end
    
    % ---------------------------------------------------------------------
    % C. PARSE PRT FILE (Robust Parser)
    % ---------------------------------------------------------------------
    fid = fopen([filename '.prt'], 'r');
    modes_found = false;
    k_list = [];
    alpha_list = [];
    
    while ~feof(fid)
        line = fgetl(fid);
        % Look for the table header
        if contains(line, 'I    k (1/m)'), modes_found = true; continue; end
        
        if modes_found
            % Read numerical data
            data = sscanf(line, '%d %f %f %f %f');
            if length(data) >= 3
                k_list(end+1) = data(2);
                alpha_list(end+1) = data(3);
            elseif isempty(line) || contains(line, '___')
                % Stop if empty line or separator line encountered
                if ~isempty(k_list), break; end 
            end
        end
    end
    fclose(fid);
    
    % Complex wavenumbers
    k_kraken = k_list + 1i * abs(alpha_list); 
    fprintf('    -> Found %d modes.\n', length(k_kraken));
    
    % ---------------------------------------------------------------------
    % D. BUILD 2D PRESSURE FIELD (Manual Sum)
    % ---------------------------------------------------------------------
    k0 = omega / cw;
    P_2D = zeros(length(z_vec), length(r_vec)); 
    
    for m = 1:length(k_kraken)
        kr = k_kraken(m);
        kz = sqrt(k0^2 - kr^2);
        
        % Bottom vertical wavenumber
        alpha_nep = alphab * freq / cb / 8.686;
        kb_c = (omega/cb) + 1i*alpha_nep;
        kzb = sqrt(kb_c^2 - kr^2);
        
        % Normalization (Same as Python code)
        term1 = D/2;
        term2 = sin(2*kz*D)/(4*kz);
        term3 = (rhow/rhob) * (sin(kz*D)^2) / (2*1i*kzb);
        ModeAmp = 1.0 / sqrt(term1 - term2 + term3);
        
        % Mode Shapes
        Psi_z = ModeAmp * sin(kz * z_vec);  
        Psi_s = ModeAmp * sin(kz * zs);     
        
        % Hankel (Far field approx)
        phase = kr * r_vec;
        Hankel_r = sqrt(2 ./ (pi * kr * r_vec)) .* exp(1i * (phase - pi/4)); 
        
        % Contribution
        contribution = (1i / (4*rhow)) * Psi_s * (Psi_z.' * Hankel_r);
        P_2D = P_2D + contribution;
    end
    
    % ---------------------------------------------------------------------
    % E. SAVE OUTPUT
    % ---------------------------------------------------------------------
    % Apply 4*pi Correction factor to match Point Source definition
    TL_2D = -20 * log10(abs(P_2D) * 4 * pi + 1e-20);
    
    % Save to unique file like 'TL_kraken_30.mat'
    out_name = sprintf('TL_kraken_2026_260122_%d.mat', round(freq));
    save(out_name, 'TL_2D', 'r_vec', 'z_vec', 'freq', 'cb', 'rhob', 'alphab');
    fprintf('    -> Saved to %s\n', out_name);
    
end

fprintf('\n[DONE] All frequencies processed.\n');