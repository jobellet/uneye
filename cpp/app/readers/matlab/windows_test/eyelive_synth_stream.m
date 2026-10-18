function s = eyelive_synth_stream(fs, seconds, seed)
% EYELIVE_SYNTH_STREAM  A synthetic eye: fixations with noise and saccades that follow a main sequence.
%   s = eyelive_synth_stream(1000, 20, 7)  ->  s.t, s.x, s.y (degrees, column vectors), s.onset (sample index of every saccade onset, 1-based), s.amp (deg), s.dur_ms
% Deterministic for a given seed (same numbers in MATLAB and Octave: it uses a small linear congruential generator, not rand).
    N = round(fs * seconds); state = double(seed) + 12345;
    function u = u01()
        state = mod(state * 1103515245 + 12345, 2147483648); u = state / 2147483648;
    end
    function g = gauss()
        a = max(u01(), 1e-12); b = u01(); g = sqrt(-2 * log(a)) * cos(2 * pi * b);
    end
    dx = zeros(N, 1); dy = zeros(N, 1); onset = []; amp = []; dur = [];
    i = round(0.5 * fs); cx = 0; cy = 0;
    while true
        i = i + round((0.35 + 0.45 * u01()) * fs);                      % fixation of 0.35 to 0.8 s
        a = 1.5 + 11 * u01() ^ 1.5; D = 20 + 2.2 * a; Dn = round(D * fs / 1000); th = 2 * pi * u01(); if hypot(cx, cy) > 8, th = atan2(-cy, -cx) + (u01() - 0.5) * pi; end      % the eye stays within about 15 degrees of the centre
        if i + Dn + 5 > N, break; end
        tau = ((1:Dn)' - 0) / Dn; pos = a * (tau - sin(2 * pi * tau) / (2 * pi)); step = [pos(1); diff(pos)];       % minimum-jerk-like position profile: bell-shaped speed
        dx(i + (1:Dn)) = step * cos(th); dy(i + (1:Dn)) = step * sin(th); onset(end + 1, 1) = i + 1; amp(end + 1, 1) = a; dur(end + 1, 1) = D; cx = cx + a * cos(th); cy = cy + a * sin(th); %#ok<AGROW>
        i = i + Dn;
    end
    nx = zeros(N, 1); ny = zeros(N, 1); for k = 1:N, nx(k) = 0.02 * gauss(); ny(k) = 0.02 * gauss(); end
    s.t = (0:N - 1)' / fs; s.x = cumsum(dx) + nx; s.y = cumsum(dy) + ny; s.onset = onset; s.amp = amp; s.dur_ms = dur; s.fs = fs;
end
