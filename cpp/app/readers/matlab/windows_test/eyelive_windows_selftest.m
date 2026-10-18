function report = eyelive_windows_selftest(bin_dir, varargin)
% EYELIVE_WINDOWS_SELFTEST  Tests EyeLive on Windows with MATLAB (R2020b or later): what the C++ tests cannot show on a CI machine.
%
%   eyelive_windows_selftest('C:\eyelive\bin')        % the folder that holds eyelive_cli.exe, eyelive_tool.exe (and eyelive_feed_probe.exe)
%
% What it does (about 2 minutes, nothing leaves the computer, only localhost ports 5005 and 6000 and a temporary folder are used):
%   A. environment: MATLAB version, timer resolution of this Windows
%   B. frame mailbox: eyelive_frames.m (the MATLAB branch with memmapfile, never run before) writes frames and candidates; they are read back byte by byte,
%      and by the C++ reader (eyelive_feed_probe.exe) when it is there
%   C. a tracker-like stream: a synthetic eye is sent by UDP at 1000 Hz to eyelive_cli.exe, which records it and fires a UDP trigger at each saccade onset;
%      MATLAB measures the latency of the triggers, reads the state block (eyelive_open.m) WHILE the session is being written by the other process,
%      and compares the detected saccades with the ones that were generated
%   D. after the recording: eyelive_tool.exe info / export; the MAT-file and the provenance journal are read back in MATLAB
% Options: 'Seconds' (stream length, default 20), 'Keep' (true: keep the temporary folder), 'Port' (UDP port of the engine, default 5005), 'TriggerPort' (default 6000).
% The text of the report is printed, returned in `report` and saved as eyelive_windows_selftest_report.txt in the current folder: send that file back.
    p = inputParser; addParameter(p, 'Seconds', 20); addParameter(p, 'Keep', false); addParameter(p, 'Port', 5005); addParameter(p, 'TriggerPort', 6000); parse(p, varargin{:}); o = p.Results;
    here = fileparts(mfilename('fullpath')); addpath(here); addpath(fileparts(here)); addpath(fullfile(fileparts(here), '..', '..', 'tools'));     % windows_test, readers/matlab (eyelive_open.m), cpp/app/tools (eyelive_frames.m)
    cli = fullfile(bin_dir, 'eyelive_cli.exe'); tool = fullfile(bin_dir, 'eyelive_tool.exe'); probe = fullfile(bin_dir, 'eyelive_feed_probe.exe');
    work = fullfile(tempdir, ['eyelive_selftest_' datestr(now, 'yyyymmdd_HHMMSS')]); mkdir(work); L = {}; fails = 0; warns = 0;
    function say(fmt, varargin), line = sprintf(fmt, varargin{:}); L{end + 1} = line; fprintf('%s\n', line); end
    function ok(cond, what, detail)
        if nargin < 3, detail = ''; end
        if cond, say('[ ok ] %s %s', what, detail); else, fails = fails + 1; say('[FAIL] %s %s', what, detail); end
    end
    function warn(cond, what, detail)
        if nargin < 3, detail = ''; end
        if ~cond, warns = warns + 1; say('[warn] %s %s', what, detail); else, say('[ ok ] %s %s', what, detail); end
    end
    say('EyeLive Windows self-test, %s', datestr(now));
    say('MATLAB %s on %s, folder %s', version, computer, bin_dir);
    ok(exist(cli, 'file') == 2, 'eyelive_cli.exe found'); ok(exist(tool, 'file') == 2, 'eyelive_tool.exe found');
    if exist(cli, 'file') ~= 2 || exist(tool, 'file') ~= 2, report = strjoin(L, newline); error('eyelive:selftest', 'eyelive_cli.exe and eyelive_tool.exe are needed in %s', bin_dir); end
    ok(exist('udpport', 'file') ~= 0, 'udpport (MATLAB R2020b or later) is available'); ok(exist('memmapfile', 'file') ~= 0, 'memmapfile is available');

    % ---------------------------------------------------------------- A. environment
    say('--- A. environment');
    t = zeros(200, 1); for k = 1:200, a = tic; pause(0.001); t(k) = toc(a) * 1000; end
    q = sort(t); say('pause(1 ms) takes %.2f ms (median), %.2f ms (95%%), %.2f ms (max): the default Windows timer step is 15.6 ms', median(t), q(ceil(0.95 * numel(q))), max(t));
    a = tic; n = 0; while toc(a) < 0.05, n = n + 1; end; say('a busy loop makes %.0f iterations per ms', n / 50);

    % ---------------------------------------------------------------- B. frame mailbox written by MATLAB
    say('--- B. frame mailbox (eyelive_frames.m, MATLAB branch)');
    mb = fullfile(work, 'stim.frames'); okB = true;
    try
        fw = eyelive_frames('open', mb, 64, 64, 3, [-20 15 20 -15]);
        img = uint8(110 * ones(64, 64, 3)); img(9:20, 45:56, 1) = 230; img(9:20, 45:56, 2) = 40; img(9:20, 45:56, 3) = 40; img(1, 1, :) = [1 2 3];       % a red square, and a marked top-left pixel
        for k = 1:5, fw = eyelive_frames('frame', fw, img, 100 + k / 30); end
        fw = eyelive_frames('candidates', fw, [10 5 1; -8 -2 0.5; 3 3 1], 12, 100.2);
        eyelive_frames('close', fw); clear fw;
    catch err
        okB = false; ok(false, 'eyelive_frames.m runs', err.message);
    end
    if okB
        m = memmapfile(mb, 'Format', 'uint8', 'Writable', false); raw = m.Data; clear m;
        u32 = @(o) typecast(raw(o + (1:4)), 'uint32'); u64 = @(o) typecast(raw(o + (1:8)), 'uint64'); f64 = @(o) typecast(raw(o + (1:8)), 'double');
        ok(strcmp(char(raw(1:8))', 'EYLFRM01'), 'magic'); ok(u32(8) == 1 && u32(12) == 256 && u32(16) == 64 && u32(20) == 64 && u32(24) == 3 && u32(28) == 3, 'header fields (version, sizes, slots)');
        ok(u64(40) == 5, 'frame_count is 5', sprintf('(%d)', u64(40))); ok(isequal(f64(48), -20) && isequal(f64(72), -15), 'gaze rectangle');
        slot = 256 + 16 * 64 + 32 + 64 * 64 * 3 + 0; first = 256 + 16 * 64; sl = @(k) first + (32 + 64 * 64 * 3) * k;      % slot k = 0, 1, 2
        s5 = sl(mod(5 - 1, 3)); ok(u64(s5 + 16) == 5 && mod(u64(s5), 2) == 0 && abs(f64(s5 + 8) - (100 + 5 / 30)) < 1e-9, 'newest slot: frame number 5, even sequence, time stamp');
        px = raw(s5 + 32 + (1:3)); ok(isequal(double(px(:))', [1 2 3]), 'pixel order: the top-left pixel is first (row-major, interleaved)'); y = 12; x = 50; off = s5 + 32 + ((y - 1) * 64 + (x - 1)) * 3; ok(isequal(double(raw(off + (1:3)))', [230 40 40]), 'a pixel inside the red square (row 12, column 50)');
        ok(u64(80) == 2 && u32(88) == 3 && abs(f64(96) - 12) < 1e-12 && abs(f64(104) - 100.2) < 1e-12, 'candidates: sequence, count, use-until, time stamp'); c = typecast(raw(256 + (1:48)), 'single'); ok(isequal(c([1 2 3 5 6 7 9 10 11]), single([10 5 1 -8 -2 0.5 3 3 1])), 'candidate values');
        if exist(probe, 'file') == 2
            [st, out] = system(sprintf('"%s" "%s" 3 --saliency', probe, mb)); ok(st == 0 && contains(out, 'explicit=1') && contains(out, 'candidates=3'), 'the C++ reader (eyelive_feed_probe.exe) sees the MATLAB file', strtrim(out(max(1, end - 160):end)));
        else, say('[skip] eyelive_feed_probe.exe not in the folder: the C++ reader was not tried'); end
    end

    % ---------------------------------------------------------------- C. a tracker-like stream
    say('--- C. UDP stream, triggers, state block');
    fs = 1000; s = eyelive_synth_stream(fs, o.Seconds, 7); N = numel(s.t); session = fullfile(work, 'stream.eyelive'); rules = fullfile(work, 'rules.txt'); logf = fullfile(work, 'cli.log');
    fid = fopen(rules, 'w'); fprintf(fid, 'name=sacc;cls=1;anchor=provisional;delay_ms=0;sink=udp:127.0.0.1:%d;payload=S {t}\n', o.TriggerPort); fclose(fid);
    rx = udpport('datagram', 'IPV4', 'LocalPort', o.TriggerPort); tx = udpport('datagram', 'IPV4');
    cmd = sprintf('start /B "" "%s" --udp %d --duration %d --session "%s" --triggers "%s" --quiet > "%s" 2>&1', cli, o.Port, round(o.Seconds) + 8, session, rules, logf); system(cmd); pause(1.5);
    send_t = nan(N, 1); trig_t = []; trig_arr = []; state_n = []; state_time = []; mm = []; per = 5; a0 = tic; k = 1; nextstate = 3;
    while k <= N
        e = min(N, k + per - 1); due = (k - 1) / fs; while toc(a0) < due, end                                          % busy wait: the Windows timer is too coarse for 5 ms
        pk = eyelive_build_packet(s.t(k:e), s.x(k:e), s.y(k:e)); write(tx, pk, 'uint8', '127.0.0.1', o.Port); send_t(k:e) = toc(a0);
        if rx.NumDatagramsAvailable > 0
            d = read(rx, rx.NumDatagramsAvailable, 'uint8'); now_s = toc(a0); for q = 1:numel(d), txt = char(d(q).Data); v = sscanf(txt, 'S %f'); if ~isempty(v), trig_t(end + 1, 1) = v; trig_arr(end + 1, 1) = now_s; end, end %#ok<AGROW>
        end
        if toc(a0) > nextstate && exist(session, 'file') == 2       % the state block of the session file, read while the engine writes it
            try
                if isempty(mm), mm = eyelive_open(session); end
                st = mm.Data; state_n(end + 1) = double(st.n_samples); state_time(end + 1) = toc(a0); %#ok<AGROW>
            catch err
                warn(false, 'reading the state block while recording', err.message); mm = []; nextstate = inf;
            end
            nextstate = nextstate + 2;
        end
        k = e + 1;
    end
    t_end = toc(a0); while toc(a0) < t_end + 1.0                                                              % triggers that are still on their way
        if rx.NumDatagramsAvailable > 0, d = read(rx, rx.NumDatagramsAvailable, 'uint8'); now_s = toc(a0); for q = 1:numel(d), v = sscanf(char(d(q).Data), 'S %f'); if ~isempty(v), trig_t(end + 1, 1) = v; trig_arr(end + 1, 1) = now_s; end, end, end %#ok<AGROW>
    end
    send_rate = N / t_end; say('%d samples sent in %.2f s (%.0f samples/s)', N, t_end, send_rate); warn(abs(send_rate - fs) < 20, 'the sender kept the 1000 Hz pace', sprintf('(%.0f /s)', send_rate));
    r = eyelive_latency_stats(trig_t, trig_arr, send_t, s.t(s.onset), fs);
    say('%d triggers received for %d generated saccades; recall %.0f %%, %d without a saccade', numel(trig_t), numel(s.onset), 100 * r.recall, r.false_triggers);
    say('latency from the SEND of the first detected sample to the ARRIVAL of the trigger: median %.1f ms, 95%% %.1f ms, max %.1f ms (includes the 5 ms packets and the UDP trip)', r.median_ms, r.p95_ms, r.max_ms);
    say('delay between the generated onset and the detected onset: median %.1f ms', median(r.detect_delay_ms(~isnan(r.detect_delay_ms))));
    ok(r.recall >= 0.9, 'at least 90 % of the saccades fired a trigger'); warn(r.median_ms < 20, 'median trigger latency below 20 ms'); warn(r.p95_ms < 40, '95 % of the triggers below 40 ms'); warn(r.false_triggers <= 2, 'at most 2 triggers without a saccade');
    if numel(state_n) >= 2, ok(all(diff(state_n) > 0), 'the state block grows while the session is written by the other process', sprintf('(%d reads, last n_samples %d)', numel(state_n), state_n(end))); else, warn(false, 'state block reads', 'fewer than 2 reads'); end
    pause(max(0, round(o.Seconds) + 10 - (toc(a0) - 0)));                                                    % the CLI stops after its --duration

    % ---------------------------------------------------------------- D. after the recording
    say('--- D. after the recording (eyelive_tool.exe)');
    clear mm; delete(rx); delete(tx);
    for tries = 1:20, if exist([session '.provenance.json'], 'file') == 2, break; end, pause(1); end
    ok(exist(session, 'file') == 2, 'session file written'); ok(exist([session '.provenance.json'], 'file') == 2, 'provenance journal written'); ok(exist([session '.report.html'], 'file') == 2, 'report written');
    if exist([session '.provenance.json'], 'file') == 2
        pj = jsondecode(fileread([session '.provenance.json'])); ok(pj.samples == N, 'the session holds every sample sent', sprintf('(%d of %d)', pj.samples, N)); say('engine version %s, detector %s, data SHA-256 %s', pj.engine_version, pj.detector_id, pj.data_sha256(1:16));
    end
    [st, out] = system(sprintf('"%s" info "%s"', tool, session)); ok(st == 0 && contains(out, 'EyeLive session report'), 'eyelive_tool info'); say('%s', strtrim(out));
    matf = fullfile(work, 'export.mat'); csvf = fullfile(work, 'events.csv'); [st, out] = system(sprintf('"%s" export "%s" --mat "%s" --csv "%s" --bids "%s"', tool, session, matf, csvf, fullfile(work, 'bids'))); ok(st == 0, 'eyelive_tool export', strtrim(out));
    if exist(matf, 'file') == 2
        D = load(matf); ok(all(isfield(D, {'t', 'x', 'y', 'valid', 'label', 'expert', 'events', 'rate_hz', 'deg_per_unit'})), 'the MAT-file holds t, x, y, valid, label, expert, events, rate_hz, deg_per_unit');
        ok(numel(D.t) == N && max(abs(D.x(1:N) - single(s.x))) < 1e-4, 'the samples in the MAT-file are those that were sent', sprintf('(max difference %.2g)', max(abs(D.x(1:N) - single(s.x)))));
        ok(size(D.events, 2) == 9 && size(D.events, 1) >= 0.9 * numel(s.onset), 'the events matrix', sprintf('(%d events, %d generated)', size(D.events, 1), numel(s.onset)));
        if ~isempty(D.events), say('saccade amplitude of the engine against the generated one: median ratio %.2f', median(D.events(1:min(end, numel(s.amp)), 6) ./ s.amp(1:min(end, size(D.events, 1)))));  end
    end
    T = readtable(csvf); ok(height(T) >= 0.9 * numel(s.onset), 'events CSV read by readtable', sprintf('(%d rows)', height(T)));
    say('--- result: %d failure(s), %d warning(s)', fails, warns);
    report = strjoin(L, newline); fid = fopen(fullfile(pwd, 'eyelive_windows_selftest_report.txt'), 'w'); fprintf(fid, '%s\n', report); fclose(fid); say('report saved in %s', fullfile(pwd, 'eyelive_windows_selftest_report.txt'));
    if ~o.Keep, try, rmdir(work, 's'); catch, end, else, say('files kept in %s', work); end
end
