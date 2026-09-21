% 1. Read the file
filename = 'mysim.shd';
[ title, type, freq, atten, Pos, p_complex ] = read_shd( filename );

% 2. Extract Vectors
r_coords = Pos.r.range; % Ranges
z_coords = Pos.r.depth; % Depths

% 3. Check for Zeros/NaN (The "Truth Test")
max_p = max(max(abs(p_complex)));
fprintf('Max Pressure Amplitude: %e\n', max_p);

if max_p == 0 || isnan(max_p)
    error('CRITICAL: The data is all zeros or NaN. The simulation failed physics-side.');
end

% 4. Plot (Quick Check)
figure;
imagesc( r_coords, z_coords, -20*log10(abs(p_complex) + 1e-12) );
colorbar; colormap(jet); set(gca, 'YDir', 'reverse');
title(['TL at ' num2str(freq) ' Hz']);
xlabel('Range (km)'); ylabel('Depth (m)');
caxis([40 100]); % Set standard dB limits

% 5. EXPORT FOR PYTHON
% We save as a .mat file, which scipy can load easily
save('bellhop_data.mat', 'p_complex', 'z_coords', 'r_coords', 'freq');
fprintf('Success! Data saved to bellhop_data.mat\n');