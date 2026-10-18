function r = eyelive_latency_stats(trig_t_stream, trig_arrival_s, send_time_of_sample, truth_onset_t, fs)
% EYELIVE_LATENCY_STATS  Latency from the moment the sample that starts a detected saccade was SENT to the moment its trigger ARRIVED, and the recall of the triggers.
%   trig_t_stream   stream time (s) in the payload of every trigger (the onset the engine detected)
%   trig_arrival_s  arrival time of every trigger (s, same clock as send_time_of_sample)
%   send_time_of_sample  send time (s) of every sample, indexed from 1
%   truth_onset_t   stream time (s) of the true saccade onsets
% Returns r.latency_ms (per trigger), r.median_ms, r.p95_ms, r.max_ms, r.recall (share of true onsets with a trigger in [-5, +80] ms), r.false_triggers, r.detect_delay_ms (detected onset minus the true onset).
    n = numel(trig_t_stream); idx = round(trig_t_stream * fs) + 1; idx = min(max(idx, 1), numel(send_time_of_sample));
    r.latency_ms = 1000 * (trig_arrival_s(:) - send_time_of_sample(idx(:))); r.latency_ms = r.latency_ms(:);
    if n == 0, r.median_ms = NaN; r.p95_ms = NaN; r.max_ms = NaN; else, q = sort(r.latency_ms); r.median_ms = q(ceil(0.5 * n)); r.p95_ms = q(min(n, ceil(0.95 * n))); r.max_ms = q(end); end
    hit = false(numel(truth_onset_t), 1); used = false(n, 1); r.detect_delay_ms = nan(numel(truth_onset_t), 1);
    for k = 1:numel(truth_onset_t)
        d = 1000 * (trig_t_stream(:) - truth_onset_t(k)); j = find(d >= -5 & d <= 80 & ~used, 1); if ~isempty(j), hit(k) = true; used(j) = true; r.detect_delay_ms(k) = d(j); end
    end
    r.recall = mean(hit); r.false_triggers = sum(~used);
end
