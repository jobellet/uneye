function results = simulate_three_targets_dataset(varargin)
% SIMULATE_THREE_TARGETS_DATASET Simulation du jeu de prediction sur saccades reelles
%
%   Ce script charge les enregistrements oculaires de U'n'Eye (dataset1 & dataset2),
%   en extrait les saccades larges (ex: amplitude >= 3 DVA), et teste
%   l'algorithme de prediction et d'esquive en temps reel:
%
%   Pour chaque essai / saccade reelle:
%   1. La trajectoire est centree sur la fixation pre-saccadique (0, 0).
%   2. Une des 3 cibles (Rouge, Verte ou Bleue) est placee EXACTEMENT a la position
%      finale reelle d'atterrissage de la saccade: P_final = (x_end, y_end).
%      Les 2 autres cibles sont disposees a la meme excentricite a +/- 120 deg.
%   3. La trajectoire est rejouee echantillon par echantillon (1000 Hz):
%      - Des que le regard quitte la fixation (> 1.5 DVA), l'algorithme calcule
%        l'alignement cosinus entre le vecteur d'amorce et les 3 cibles candidates.
%      - La cible avec l'alignement maximal est predite.
%      - Esquive: la cible predite est decalee vers l'exterieur (+5.0 DVA).
%   4. Evaluation a l'atterrissage:
%      - Distance a la fovea (< 2.0 DVA).
%      - Si l'esquive a reussi, le sujet atterrit a 5.0 DVA de la cible decalee (0 pt).
%      - Si l'algorithme s'est trompe, la cible n'a pas bouge (< 2 DVA -> +10 pts).
%
%   Parametres (Nom / Valeur):
%     'MinAmp'      - Amplitude minimale des saccades en DVA (defaut: 3.0)
%     'ShiftDist'   - Distance de decalage / esquive en DVA (defaut: 5.0)
%     'FoveaRadius' - Rayon foveal de score en DVA (defaut: 2.0)
%     'TriggerDist' - Deplacement seuil pour declencher la prediction (defaut: 1.5)
%     'MaxTrials'   - Nombre max d'essais a tester (defaut: inf = tous)

    p = inputParser;
    addParameter(p, 'MinAmp', 3.0, @isnumeric);
    addParameter(p, 'ShiftDist', 5.0, @isnumeric);
    addParameter(p, 'FoveaRadius', 2.0, @isnumeric);
    addParameter(p, 'TriggerDist', 1.5, @isnumeric);
    addParameter(p, 'MaxTrials', inf, @isnumeric);
    parse(p, varargin{:});
    opts = p.Results;

    % Localisation du dossier data/
    here = fileparts(mfilename('fullpath'));
    curr = here;
    data_dir = '';
    for k = 1:5
        cand = fullfile(curr, 'data');
        if exist(cand, 'dir')
            data_dir = cand;
            break;
        end
        curr = fileparts(curr);
    end
    if isempty(data_dir)
        data_dir = 'C:\Users\joach\Documents\GitHub\uneye\data';
    end

    fprintf('=== SIMULATION DU JEU SUR SACCADES REELLES (U''n''Eye) ===\n');
    fprintf('Dossier de donnees : %s\n', data_dir);
    fprintf('Filtre : Amplitude >= %.1f DVA | Esquive = %.1f DVA | Seuil Fovea = %.1f DVA\n\n', ...
        opts.MinAmp, opts.ShiftDist, opts.FoveaRadius);

    datasets = {'dataset1', 'dataset2'};
    splits = {'setA', 'setB'};

    % Extraction des saccades
    saccades = {};
    for d = 1:numel(datasets)
        ds = datasets{d};
        for s = 1:numel(splits)
            sp = splits{s};
            x_file = dir(fullfile(data_dir, ds, sprintf('*_X_%s.csv', sp)));
            y_file = dir(fullfile(data_dir, ds, sprintf('*_Y_%s.csv', sp)));
            l_file = dir(fullfile(data_dir, ds, sprintf('*_Labels_%s.csv', sp)));

            if isempty(x_file) || isempty(y_file) || isempty(l_file), continue; end

            X = readmatrix(fullfile(data_dir, ds, x_file(1).name));
            Y = readmatrix(fullfile(data_dir, ds, y_file(1).name));
            L = readmatrix(fullfile(data_dir, ds, l_file(1).name));

            n_trials = min([size(X, 1), size(Y, 1), size(L, 1)]);
            for r = 1:n_trials
                row_l = L(r, :);
                row_x = X(r, :);
                row_y = Y(r, :);

                in_s = false;
                s_idx = 1;
                for i = 1:numel(row_l)
                    if row_l(i) == 1 && ~in_s
                        in_s = true;
                        s_idx = i;
                    elseif row_l(i) == 0 && in_s
                        in_s = false;
                        e_idx = i - 1;
                        dx = row_x(e_idx) - row_x(s_idx);
                        dy = row_y(e_idx) - row_y(s_idx);
                        amp = hypot(dx, dy);

                        if amp >= opts.MinAmp
                            t_pre = max(1, s_idx - 100);
                            t_post = min(numel(row_x), e_idx + 150);
                            sacc.dataset = ds;
                            sacc.amp = amp;
                            sacc.duration_ms = e_idx - s_idx + 1;
                            sacc.x = row_x(t_pre:t_post);
                            sacc.y = row_y(t_pre:t_post);
                            sacc.onset = s_idx - t_pre + 1;
                            sacc.offset = e_idx - t_pre + 1;
                            saccades{end+1} = sacc; %#ok<AGROW>

                            if numel(saccades) >= opts.MaxTrials, break; end
                        end
                    end
                end
                if numel(saccades) >= opts.MaxTrials, break; end
            end
            if numel(saccades) >= opts.MaxTrials, break; end
        end
        if numel(saccades) >= opts.MaxTrials, break; end
    end

    n_tot = numel(saccades);
    fprintf('-> %d saccades larges extraites (amplitude max: %.2f DVA)\n\n', n_tot, max(cellfun(@(c) c.amp, saccades)));

    % Variables d'accumulation
    latencies = [];
    frac_amplitudes = [];
    landing_distances = [];
    correct_predictions = 0;
    shifted_count = 0;
    subject_hits = 0;
    trials_log = repmat(struct(), n_tot, 1);

    for idx = 1:n_tot
        s = saccades{idx};
        x = s.x;
        y = s.y;
        onset = s.onset;
        offset = s.offset;

        % Baseline de fixation
        w_pre = max(1, onset - 40);
        x0 = mean(x(w_pre:onset));
        y0 = mean(y(w_pre:onset));

        % Position d'atterrissage finale
        w_post = min(numel(x), offset + 40);
        x_end = mean(x(offset:w_post));
        y_end = mean(y(offset:w_post));

        xc = x - x0;
        yc = y - y0;
        p_final = [x_end - x0, y_end - y0];
        r_final = norm(p_final);
        theta_final = atan2(p_final(2), p_final(1));

        % Placement des 3 cibles : la cible chosen_k est a P_final
        chosen_k = mod(idx - 1, 3) + 1; % 1, 2, ou 3
        targets_orig = zeros(3, 2);
        for i = 1:3
            ang = theta_final + (i - chosen_k) * (2 * pi / 3);
            targets_orig(i, :) = [r_final * cos(ang), r_final * sin(ang)];
        end
        targets_curr = targets_orig;

        % Rejeu pas a pas (1000 Hz)
        predicted_idx = 0;
        pred_latency = 0;
        pred_frac = 0;

        for t = onset:offset
            gx = xc(t);
            gy = yc(t);
            dist_c = hypot(gx, gy);

            if dist_c >= opts.TriggerDist && predicted_idx == 0
                vec = [gx, gy];
                cos_sims = zeros(3, 1);
                for i = 1:3
                    ti = targets_orig(i, :);
                    cos_sims(i) = dot(vec, ti) / (norm(vec) * norm(ti));
                end
                [max_sim, best_i] = max(cos_sims);
                if max_sim > 0.50
                    predicted_idx = best_i;
                    pred_latency = t - onset;
                    pred_frac = dist_c / r_final;

                    % Esquive : decaler la cible vers l'exterieur
                    out_dir = targets_orig(predicted_idx, :) / norm(targets_orig(predicted_idx, :));
                    targets_curr(predicted_idx, :) = targets_orig(predicted_idx, :) + out_dir * opts.ShiftDist;
                    break;
                end
            end
        end

        if predicted_idx > 0
            shifted_count = shifted_count + 1;
            latencies(end+1) = pred_latency; %#ok<AGROW>
            frac_amplitudes(end+1) = pred_frac; %#ok<AGROW>
            if predicted_idx == chosen_k
                correct_predictions = correct_predictions + 1;
            end
        end

        % Evaluation a l'atterrissage
        dists = zeros(3, 1);
        for i = 1:3
            dists(i) = norm(p_final - targets_curr(i, :));
        end
        [min_dist, closest_target] = min(dists);
        landing_distances(end+1) = min_dist; %#ok<AGROW>

        is_hit = (min_dist < opts.FoveaRadius);
        if is_hit
            subject_hits = subject_hits + 1;
        end

        trials_log(idx).trial = idx;
        trials_log(idx).dataset = s.dataset;
        trials_log(idx).amplitude = s.amp;
        trials_log(idx).duration_ms = s.duration_ms;
        trials_log(idx).intended_target = chosen_k;
        trials_log(idx).predicted_target = predicted_idx;
        trials_log(idx).prediction_correct = (predicted_idx == chosen_k);
        trials_log(idx).prediction_latency_ms = pred_latency;
        trials_log(idx).frac_at_trigger = pred_frac;
        trials_log(idx).min_landing_dist_dva = min_dist;
        trials_log(idx).closest_target = closest_target;
        trials_log(idx).hit = is_hit;
        trials_log(idx).points = is_hit * 10;
    end

    acc_pct = (correct_predictions / max(1, shifted_count)) * 100;
    shift_rate = (shifted_count / n_tot) * 100;
    dodge_rate = ((n_tot - subject_hits) / n_tot) * 100;
    hit_rate = (subject_hits / n_tot) * 100;

    fprintf('===================================================================\n');
    fprintf('  RESULTATS DE LA SIMULATION (MATLAB)\n');
    fprintf('===================================================================\n');
    fprintf('Saccades testees (amp >= %.1f DVA)   : %d\n', opts.MinAmp, n_tot);
    fprintf('Detection et esquive declenchees     : %d / %d (%.1f%%)\n', shifted_count, n_tot, shift_rate);
    fprintf('Precision de prediction de la cible  : %d / %d (%.1f%%)\n', correct_predictions, shifted_count, acc_pct);
    fprintf('Latence moyenne de prediction        : %.1f ms apres onset\n', mean(latencies));
    fprintf('Progression moyenne au trigger       : %.1f%% de l''amplitude\n', mean(frac_amplitudes) * 100);
    fprintf('Distance moyenne a l''atterrissage   : %.2f DVA (seuil: < %.1f DVA)\n', mean(landing_distances), opts.FoveaRadius);
    fprintf('Taux d''esquive reussie (Dodge Rate)  : %.1f%%\n', dodge_rate);
    fprintf('Points marques par le sujet (< 2 DVA): %d / %d (%.1f%%) | Score: %d pts\n', ...
        subject_hits, n_tot, hit_rate, subject_hits * 10);
    fprintf('===================================================================\n\n');

    results.num_saccades = n_tot;
    results.shift_triggered = shifted_count;
    results.prediction_accuracy_pct = acc_pct;
    results.mean_latency_ms = mean(latencies);
    results.mean_saccade_progress_pct = mean(frac_amplitudes) * 100;
    results.mean_landing_dist_dva = mean(landing_distances);
    results.dodge_success_pct = dodge_rate;
    results.subject_hits = subject_hits;
    results.total_score = subject_hits * 10;
    results.trials = trials_log;

    out_name = sprintf('results_simulation_dataset_%s.mat', datestr(now, 'yyyymmdd_HHMMSS'));
    save(out_name, 'results');
    fprintf('Resultats sauvegardes dans : %s\n', out_name);
end
