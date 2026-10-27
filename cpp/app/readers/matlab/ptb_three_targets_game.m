function results = ptb_three_targets_game(varargin)
% PTB_THREE_TARGETS_GAME Experience Psychtoolbox avec prediction du regard en temps reel
%
%   Ce jeu experimental teste la saccade et la prediction du regard en temps reel:
%   1. Le sujet fixe une croix centrale.
%   2. Trois cibles colorees (Rouge, Verte, Bleue) apparaissent a egale distance (ex: 7 DVA).
%   3. Le sujet choisit librement de regarder l'une des trois cibles.
%   4. Dès le debut de la saccade, l'algorithme predit en temps reel vers quelle cible
%      se dirige le regard (via EyeLive pred_x/pred_y ou la direction de la saccade)
%      et tente de l'esquiver en DECALANT le point choisi vers l'exterieur.
%   5. Regle du score: Si le sujet parvient a faire entrer sa fovea a MOINS DE 2 DVA
%      d'un des trois points (en feintant l'algorithme ou par correction rapide),
%      il gagne des points (+10 pts) !
%
%   Utilisation:
%     ptb_three_targets_game                     % Mode auto (EyeLive si present, sinon Souris)
%     ptb_three_targets_game('Source', 'mouse')   % Mode simulation a la souris
%     ptb_three_targets_game('Source', 'eyelive', 'Session', 'stream.eyelive')
%     ptb_three_targets_game('NumTrials', 10, 'Windowed', true)
%
%   Parametres optionnels (Nom / Valeur):
%     'NumTrials'       - Nombre d'essais (defaut: 12)
%     'Source'          - 'auto', 'eyelive', ou 'mouse' (defaut: 'auto')
%     'Session'         - Chemin du fichier de session EyeLive (defaut: 'stream.eyelive')
%     'Windowed'        - true pour fenetre de developpement, false pour plein ecran (defaut: true)
%     'FoveaRadiusDVA'  - Rayon de capture foveale (defaut: 2.0 DVA)
%     'TargetEccDVA'    - Excentricite des cibles (defaut: 7.0 DVA)
%     'ShiftDistDVA'    - Distance de decalage / esquive (defaut: 5.0 DVA)
%     'ScreenWidthCm'   - Largeur de l'ecran en cm (defaut: 53.0)
%     'ScreenDistCm'    - Distance sujet-ecran en cm (defaut: 60.0)

    % -------------------------------------------------------------------------
    % 1. Analyse des options et configuration
    % -------------------------------------------------------------------------
    p = inputParser;
    addParameter(p, 'NumTrials', 12, @isnumeric);
    addParameter(p, 'Source', 'auto', @(s) any(validatestring(s, {'auto', 'eyelive', 'mouse'})));
    addParameter(p, 'Session', 'stream.eyelive', @ischar);
    addParameter(p, 'Windowed', true, @islogical);
    addParameter(p, 'FoveaRadiusDVA', 2.0, @isnumeric);
    addParameter(p, 'TargetEccDVA', 7.0, @isnumeric);
    addParameter(p, 'ShiftDistDVA', 5.0, @isnumeric);
    addParameter(p, 'ScreenWidthCm', 53.0, @isnumeric);
    addParameter(p, 'ScreenDistCm', 60.0, @isnumeric);
    addParameter(p, 'TestMode', false, @islogical);
    parse(p, varargin{:});
    opts = p.Results;

    % Configuration des chemins (EyeLive readers + Psychtoolbox)
    here = fileparts(mfilename('fullpath'));
    addpath(here);
    tools_dir = fullfile(here, '..', '..', 'tools');
    if exist(tools_dir, 'dir'), addpath(tools_dir); end

    % Verification Psychtoolbox
    ptb_root = 'C:\Users\joach\Documents\MATLAB\Psychtoolbox-3.0.19\Psychtoolbox';
    if exist(ptb_root, 'dir')
        addpath(fullfile(ptb_root, 'PsychBasic', 'MatlabWindowsFilesR2007a'));
        addpath(genpath(ptb_root));
        addpath(fullfile(ptb_root, 'PsychBasic', 'MatlabWindowsFilesR2007a'));
    end

    if exist('Screen', 'file') == 0
        error('ptb_game:no_ptb', 'Psychtoolbox Screen.mex est introuvable. Verifiez l installation.');
    end

    % Determination de la source de regard (EyeLive ou Souris)
    source_type = opts.Source;
    mm = [];
    if strcmp(source_type, 'auto') || strcmp(source_type, 'eyelive')
        if exist(opts.Session, 'file') == 2
            try
                mm = eyelive_open(opts.Session);
                source_type = 'eyelive';
                fprintf('[EyeLive] Connecte a la session: %s\n', opts.Session);
            catch err
                warning('ptb_game:open_failed', 'Impossible d ouvrir la session EyeLive: %s', err.message);
                if strcmp(opts.Source, 'eyelive'), rethrow(err); end
                source_type = 'mouse';
            end
        else
            if strcmp(opts.Source, 'eyelive')
                error('ptb_game:file_not_found', 'Fichier EyeLive introuvable: %s', opts.Session);
            end
            source_type = 'mouse';
        end
    end
    fprintf('[Game] Mode de controle: %s\n', upper(source_type));

    % -------------------------------------------------------------------------
    % 2. Initialisation Psychtoolbox & Geometrie Visuelle (DVA <-> Pixels)
    % -------------------------------------------------------------------------
    PsychDefaultSetup(2);
    Screen('Preference', 'SkipSyncTests', 2);
    Screen('Preference', 'VisualDebugLevel', 1);
    Screen('Preference', 'SuppressAllWarnings', 1);

    screens = Screen('Screens');
    screenNumber = max(screens);

    bg_color = [30 30 35];     % Fond sombre agreable
    white = [255 255 255];
    gray  = [140 140 140];
    gold  = [255 215 0];

    % Cibles colorees: Rouge, Vert, Bleu
    colors = { ...
        [235, 65, 65],  ... % 1: Rouge
        [55, 215, 75],  ... % 2: Vert
        [60, 140, 255]  ... % 3: Bleu
    };
    color_names = {'ROUGE', 'VERT', 'BLEU'};

    if opts.Windowed
        [win, winRect] = Screen('OpenWindow', screenNumber, bg_color, [0 0 1024 768]);
    else
        [win, winRect] = Screen('OpenWindow', screenNumber, bg_color);
    end

    [screenWidthPx, screenHeightPx] = Screen('WindowSize', win);
    cx = screenWidthPx / 2;
    cy = screenHeightPx / 2;

    % Conversion DVA <-> Pixels
    % 1 degre = screenDistCm * tan(1 deg) en cm
    px_per_cm = screenWidthPx / opts.ScreenWidthCm;
    deg_per_cm = atan2d(1, opts.ScreenDistCm);
    ppd = px_per_cm / deg_per_cm;   % Pixels Per Degree of visual angle

    fovea_radius_px = opts.FoveaRadiusDVA * ppd;
    target_ecc_px   = opts.TargetEccDVA * ppd;
    target_radius_px = 0.8 * ppd;
    shift_dist_px   = opts.ShiftDistDVA * ppd;

    Screen('BlendFunction', win, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');
    Screen('TextSize', win, 24);

    % Structure de resultat
    trial_data = repmat(struct( ...
        'trial', 0, ...
        'target_chosen', 0, ...
        'target_predicted', 0, ...
        'shifted', false, ...
        'landing_x', 0, ...
        'landing_y', 0, ...
        'min_distance_dva', 0, ...
        'hit', false, ...
        'points', 0, ...
        'rt_ms', 0 ...
    ), opts.NumTrials, 1);

    total_score = 0;

    % -------------------------------------------------------------------------
    % 3. Ecran d'accueil & Instructions
    % -------------------------------------------------------------------------
    DrawFormattedText(win, double('JEU DU REGARD : ATTRAPEZ LE POINT !'), 'center', cy - 140, gold);
    msg = [ ...
        'Trois points colores (Rouge, Vert, Bleu) vont apparaitre.\n\n' ...
        'Regardez librement L''UN des trois points !\n\n' ...
        'L''algorithme essaie en direct de deviner votre cible et de l''esquiver.\n' ...
        'Si votre fovea atterrit a MOINS DE 2 DVA d''un point, vous marquez 10 points !\n\n' ...
        'Appuyez sur ESPACE pour commencer (ESC pour quitter)...' ...
    ];
    DrawFormattedText(win, double(msg), 'center', cy - 50, white);
    Screen('Flip', win);

    % Attente de la touche ESPACE
    if ~opts.TestMode
        while true
            [keyIsDown, ~, keyCode] = KbCheck;
            if keyIsDown
                if keyCode(KbName('space')), break; end
                if keyCode(KbName('ESCAPE')), Screen('CloseAll'); return; end
            end
            WaitSecs(0.01);
        end
    else
        WaitSecs(0.05);
    end

    % -------------------------------------------------------------------------
    % 4. Boucle Principale des Essais
    % -------------------------------------------------------------------------
    for t = 1:opts.NumTrials
        trial_data(t).trial = t;

        % Positions de depart des 3 cibles (triangle equilateral)
        % Angle de base aleatoire pour varier l orientation a chaque essai
        base_angle = rand * 2 * pi;
        angles = base_angle + [0, 2*pi/3, 4*pi/3];

        % Coordonnees initiales des cibles [x, y]
        targets_orig = zeros(3, 2);
        for i = 1:3
            targets_orig(i, 1) = cx + target_ecc_px * cos(angles(i));
            targets_orig(i, 2) = cy + target_ecc_px * sin(angles(i));
        end
        targets_curr = targets_orig;

        % --- Phase A: Fixation Centrale Obligatoire ---
        fixation_held_start = 0;
        fixation_req_duration = 0.400; % 400 ms de fixation stable
        while true
            % Lecture du regard / souris
            [gx, gy] = get_gaze_position(win, source_type, mm, cx, cy, ppd, opts.TestMode, 0, targets_orig);
            dist_to_center_dva = hypot(gx - cx, gy - cy) / ppd;

            if dist_to_center_dva <= 1.5
                if fixation_held_start == 0
                    fixation_held_start = GetSecs;
                elseif (GetSecs - fixation_held_start) >= fixation_req_duration
                    break; % Fixation validee !
                end
            else
                fixation_held_start = 0;
            end

            % Dessin de la croix de fixation centrale
            draw_header(win, t, opts.NumTrials, total_score, white, gold);
            draw_cross(win, cx, cy, 0.4 * ppd, [200 200 200]);
            DrawFormattedText(win, double('Fixez la croix...'), 'center', cy + 50, gray);
            Screen('Flip', win);

            % Verification ESC
            [kd, ~, kc] = KbCheck;
            if kd && kc(KbName('ESCAPE')), break; end
            WaitSecs(0.005);
        end

        % Bip sonore d apparition des cibles
        try, Beeper(600, 0.05, 0.08); catch, end

        % --- Phase B: Affichage des Cibles & Prediction en Direct ---
        t_stim_onset = GetSecs;
        target_shifted = false;
        predicted_target = 0;
        shift_vector = [0 0];

        saccade_started = false;
        t_saccade_start = 0;
        gaze_at_onset = [gx, gy];

        % Boucle de vol saccadique (max 1.5 seconde par essai)
        while (GetSecs - t_stim_onset) < 1.5
            [gx, gy, in_event, pred_pt, pred_valid] = get_gaze_position(win, source_type, mm, cx, cy, ppd, opts.TestMode, t_stim_onset, targets_orig);
            dist_from_fix_dva = hypot(gx - cx, gy - cy) / ppd;

            % Detection du declenchement de la saccade
            if ~saccade_started && (dist_from_fix_dva > 1.8 || in_event)
                saccade_started = true;
                t_saccade_start = GetSecs;
            end

            % --- ALGORITHME DE PREDICTION EN TEMPS REEL ---
            if saccade_started && ~target_shifted
                % 1. Si EyeLive fournit une prediction de point d'atterrissage (LandingNet)
                if pred_valid && ~any(isnan(pred_pt))
                    % Cible la plus proche du point predit par le reseau
                    dists = hypot(targets_curr(:, 1) - pred_pt(1), targets_curr(:, 2) - pred_pt(2));
                    [~, predicted_target] = min(dists);
                else
                    % 2. Prediction cinematique: vecteur de direction depuis la fixation
                    sacc_vec = [gx - cx, gy - cy];
                    if norm(sacc_vec) > 0.01
                        dot_prods = zeros(3, 1);
                        for i = 1:3
                            target_vec = [targets_orig(i, 1) - cx, targets_orig(i, 2) - cy];
                            % Cosine similarity (alignement angulaire)
                            dot_prods(i) = (sacc_vec * target_vec') / (norm(sacc_vec) * norm(target_vec));
                        end
                        [max_align, best_idx] = max(dot_prods);
                        if max_align > 0.50 % Vise clairement dans le cone d'un des stimuli
                            predicted_target = best_idx;
                        end
                    end
                end

                % Si une cible est predite: on la DECALE pour esquiver le regard !
                if predicted_target > 0
                    orig_pt = targets_orig(predicted_target, :);
                    % Vecteur d'esquive vers l'exterieur
                    outward_dir = (orig_pt - [cx, cy]) / norm(orig_pt - [cx, cy]);
                    shift_vector = outward_dir * shift_dist_px;
                    targets_curr(predicted_target, :) = orig_pt + shift_vector;
                    target_shifted = true;
                    fprintf('Trial %d: Cible %s predite -> DECALEE de %.1f DVA!\n', ...
                        t, color_names{predicted_target}, opts.ShiftDistDVA);
                end
            end

            % Condition d'atterrissage de la saccade:
            % La saccade a debute et le regard se stabilise ou atteint l excentricite
            if saccade_started && (dist_from_fix_dva >= 4.0 || (GetSecs - t_saccade_start) > 0.180)
                % Petite pause pour laisser la fixation terminale s'etablir (100 ms)
                WaitSecs(0.100);
                [gx, gy] = get_gaze_position(win, source_type, mm, cx, cy, ppd, opts.TestMode, t_stim_onset, targets_orig);
                break;
            end

            % --- Rendu Graphique des Cibles ---
            draw_header(win, t, opts.NumTrials, total_score, white, gold);

            % Dessin des 3 cibles
            for i = 1:3
                tx = targets_curr(i, 1);
                ty = targets_curr(i, 2);
                col = colors{i};
                rect = [tx - target_radius_px, ty - target_radius_px, tx + target_radius_px, ty + target_radius_px];
                Screen('FillOval', win, col, rect);

                % Si la cible a ete decalee, afficher l'ancienne position en filigrane fantome
                if target_shifted && i == predicted_target
                    ghost_x = targets_orig(i, 1);
                    ghost_y = targets_orig(i, 2);
                    ghost_rect = [ghost_x - target_radius_px, ghost_y - target_radius_px, ghost_x + target_radius_px, ghost_y + target_radius_px];
                    Screen('FrameOval', win, [col 100], ghost_rect, 2);
                end
            end

            % Affichage du regard en direct (pour feedback fluide)
            draw_gaze_cursor(win, gx, gy, 6, [255 255 255 180]);

            Screen('Flip', win);
            WaitSecs(0.002);
        end

        % --- Phase C: Evaluation de la Fovea (< 2 DVA) et Attribution du Score ---
        final_gaze = [gx, gy];
        dists_to_curr_targets_px = hypot(targets_curr(:, 1) - final_gaze(1), targets_curr(:, 2) - final_gaze(2));
        dists_to_curr_targets_dva = dists_to_curr_targets_px / ppd;

        [min_dist_dva, closest_target_idx] = min(dists_to_curr_targets_dva);

        % Regle: fovea a moins de 2 DVA d'un des points
        is_hit = (min_dist_dva < opts.FoveaRadiusDVA);

        if is_hit
            points_won = 10;
            total_score = total_score + points_won;
            feedback_color = [60 230 80];
            try, Beeper(880, 0.15, 0.20); catch, end
            if target_shifted && closest_target_idx == predicted_target
                verdict_txt = 'TOUCHÉ MALGRÉ L''ESQUIVE ! (+10 pts)';
            else
                verdict_txt = 'TOUCHÉ ! FEINTE RÉUSSIE (+10 pts)';
            end
        else
            points_won = 0;
            feedback_color = [240 80 80];
            try, Beeper(250, 0.15, 0.20); catch, end
            if target_shifted
                verdict_txt = sprintf('ESQUIVÉ ! L''algorithme a décalé le point %s (0 pt)', color_names{predicted_target});
            else
                verdict_txt = 'HORS CIBLE (0 pt)';
            end
        end

        % Enregistrement des donnees de l'essai
        trial_data(t).target_chosen = closest_target_idx;
        trial_data(t).target_predicted = predicted_target;
        trial_data(t).shifted = target_shifted;
        trial_data(t).landing_x = (final_gaze(1) - cx) / ppd;
        trial_data(t).landing_y = (final_gaze(2) - cy) / ppd;
        trial_data(t).min_distance_dva = min_dist_dva;
        trial_data(t).hit = is_hit;
        trial_data(t).points = points_won;
        trial_data(t).rt_ms = (t_saccade_start - t_stim_onset) * 1000;

        % --- Affichage du Feedback de Fin d'Essai (1 seconde) ---
        t_feedback = GetSecs;
        while (GetSecs - t_feedback) < 1.00
            draw_header(win, t, opts.NumTrials, total_score, white, gold);

            % Dessin des cibles
            for i = 1:3
                tx = targets_curr(i, 1);
                ty = targets_curr(i, 2);
                col = colors{i};
                rect = [tx - target_radius_px, ty - target_radius_px, tx + target_radius_px, ty + target_radius_px];
                Screen('FillOval', win, col, rect);

                % Cercle de fovea 2 DVA autour de la cible
                Screen('FrameOval', win, [col 60], [tx - fovea_radius_px, ty - fovea_radius_px, tx + fovea_radius_px, ty + fovea_radius_px], 1);
            end

            % Position finale du regard
            draw_gaze_cursor(win, final_gaze(1), final_gaze(2), 10, feedback_color);

            % Message de feedback
            DrawFormattedText(win, double(verdict_txt), 'center', cy + 90, feedback_color);
            dist_str = sprintf('Distance la plus proche: %.2f DVA (Seuil: < 2.0 DVA)', min_dist_dva);
            DrawFormattedText(win, double(dist_str), 'center', cy + 130, gray);

            Screen('Flip', win);
            WaitSecs(0.01);
        end

        % Pause inter-essai (800 ms)
        Screen('FillRect', win, bg_color);
        Screen('Flip', win);
        WaitSecs(0.400);
    end

    % -------------------------------------------------------------------------
    % 5. Ecran de Bilan & Resultats
    % -------------------------------------------------------------------------
    hits_total = sum([trial_data.hit]);
    acc_pct = (hits_total / opts.NumTrials) * 100;

    Screen('FillRect', win, bg_color);
    DrawFormattedText(win, double('PARTIE TERMINEE !'), 'center', cy - 120, gold);
    summary_txt = sprintf([ ...
        'Score Final : %d points\n\n' ...
        'Cibles atteintes (< 2 DVA) : %d / %d  (%.1f%%%%)\n\n' ...
        'Appuyez sur une touche pour quitter...' ...
    ], total_score, hits_total, opts.NumTrials, acc_pct);
    DrawFormattedText(win, double(summary_txt), 'center', cy - 40, white);
    Screen('Flip', win);

    if ~opts.TestMode
        KbStrokeWait;
    else
        WaitSecs(0.05);
    end
    Screen('CloseAll');

    results = struct( ...
        'score', total_score, ...
        'num_trials', opts.NumTrials, ...
        'hits', hits_total, ...
        'accuracy_pct', acc_pct, ...
        'trials', trial_data ...
    );

    % Sauvegarde automatique des resultats dans un fichier .mat
    out_name = sprintf('results_ptb_game_%s.mat', datestr(now, 'yyyymmdd_HHMMSS'));
    save(out_name, 'results');
    fprintf('\n[Resultats] Score: %d | Succes: %.1f%% | Fichier sauvegarde: %s\n', total_score, acc_pct, out_name);
end

% -----------------------------------------------------------------------------
% Fonctions d'assistance internes
% -----------------------------------------------------------------------------

function [gx, gy, in_event, pred_pt, pred_valid] = get_gaze_position(win, src, mm, cx, cy, ppd, test_mode, t_stim_onset, targets_orig)
% Recupere la position du regard en pixels a l ecran
    if nargin < 7, test_mode = false; end
    if nargin < 8, t_stim_onset = 0; end
    if nargin < 9, targets_orig = []; end

    in_event = false;
    pred_pt = [NaN, NaN];
    pred_valid = false;

    if test_mode
        % En TestMode, simule une fixation initiale puis une saccade fluide vers la Cible 2
        if t_stim_onset == 0
            gx = cx; gy = cy;
        else
            elapsed = GetSecs - t_stim_onset;
            if elapsed < 0.120
                gx = cx; gy = cy; in_event = false;
            else
                % Mouvement saccadique vers la cible 2
                t2 = targets_orig(2, :);
                frac = min(1.0, (elapsed - 0.120) / 0.040);
                gx = cx + frac * (t2(1) - cx);
                gy = cy + frac * (t2(2) - cy);
                in_event = (frac < 1.0);
                pred_pt = t2;
                pred_valid = true;
            end
        end
        return;
    end

    if strcmp(src, 'eyelive') && ~isempty(mm)
        [st, ok] = eyelive_state(mm, 5);
        if ok
            gx = cx + double(st.x) * ppd;
            gy = cy + double(st.y) * ppd;
            in_event = (st.in_event == 1);
            if st.pred_valid == 1
                pred_pt = [cx + double(st.pred_x) * ppd, cy + double(st.pred_y) * ppd];
                pred_valid = true;
            end
            return;
        end
    end

    % Repli automatique sur la souris
    [mx, my] = GetMouse(win);
    gx = mx;
    gy = my;
end

function draw_header(win, trial, num_trials, score, text_col, score_col)
    t_txt = sprintf('Essai %d / %d', trial, num_trials);
    s_txt = sprintf('Score: %d pts', score);
    Screen('DrawText', win, t_txt, 30, 25, text_col);
    Screen('DrawText', win, s_txt, 850, 25, score_col);
end

function draw_cross(win, x, y, len, col)
    Screen('DrawLine', win, col, x - len, y, x + len, y, 2);
    Screen('DrawLine', win, col, x, y - len, x, y + len, 2);
end

function draw_gaze_cursor(win, x, y, r, col)
    Screen('FillOval', win, col, [x - r, y - r, x + r, y + r]);
end
