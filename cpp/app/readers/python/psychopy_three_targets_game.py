#!/usr/bin/env python3
"""
PsychoPy Three-Target Gaze Prediction Game (Equivalent to PTB_THREE_TARGETS_GAME)
--------------------------------------------------------------------------------
Experimental eye-tracking game testing real-time saccade landing prediction:
1. Participant fixates central cross (< 1.5 DVA for >= 400 ms).
2. Three colored stimuli (Red, Green, Blue) appear at equal eccentricity (7.0 DVA).
3. The participant freely chooses to look at one of the three targets.
4. In real-time, the algorithm predicts which target is being targeted:
   - Via EyeLive LandingNet prediction (pred_x, pred_y, pred_valid), OR
   - Via early saccadic trajectory angle (cosine similarity with target vectors).
5. As soon as predicted, the algorithm dodges by shifting the chosen target outward (+5.0 DVA).
6. Scoring: If the participant's fovea lands within < 2.0 DVA of any target,
   the participant wins 10 points (+10 pts)!

Works both with real eye-tracker (via EyeLive shared memory) and mouse simulation.
Can run in headless test mode (--test-mode) for CI/validation.
"""

import sys
import os
import math
import time
import json
import random
import argparse
from datetime import datetime

# Add reader path for StateReader
HERE = os.path.dirname(os.path.abspath(__file__))
READER_DIR = os.path.join(HERE, "..", "cpp", "app", "readers", "python")
if os.path.isdir(READER_DIR):
    sys.path.insert(0, READER_DIR)

try:
    from eyelive_reader import StateReader
except ImportError:
    StateReader = None


def compute_geometry(screen_width_px, screen_height_px, screen_width_cm, screen_dist_cm):
    """Compute pixels-per-degree (PPD) and visual angle conversions."""
    px_per_cm = screen_width_px / screen_width_cm
    deg_per_cm = math.degrees(math.atan2(1.0, screen_dist_cm))
    ppd = px_per_cm / deg_per_cm
    return ppd


class MockPsychoPyWindow:
    """Mock window for headless CI or environments without PsychoPy display."""
    def __init__(self, size=(1024, 768)):
        self.size = size
        self.is_mock = True

    def flip(self):
        pass

    def close(self):
        pass


def run_headless_simulation(num_trials=2, fovea_radius_dva=2.0, target_ecc_dva=7.0, shift_dist_dva=5.0):
    """Run full experiment logic in headless mode for automated verification."""
    print("[Headless] Running three-target prediction game simulation...")
    ppd = 35.0  # Approx standard PPD
    target_names = ["ROUGE", "VERT", "BLEU"]
    trials_data = []
    total_score = 0
    hits = 0

    for t in range(1, num_trials + 1):
        base_angle = random.random() * 2 * math.pi
        angles = [base_angle + i * (2 * math.pi / 3) for i in range(3)]
        targets_orig = [(target_ecc_dva * math.cos(a), target_ecc_dva * math.sin(a)) for a in angles]
        targets_curr = list(targets_orig)

        # Fixation phase
        # Saccade to target 1 (VERT)
        chosen_target_idx = 1
        target_pt = targets_orig[chosen_target_idx]

        # Early trajectory detection: angle points to target 1
        predicted_idx = 1
        # Shift target 1 outward
        out_angle = math.atan2(target_pt[1], target_pt[0])
        shifted_pt = (
            target_pt[0] + shift_dist_dva * math.cos(out_angle),
            target_pt[1] + shift_dist_dva * math.sin(out_angle),
        )
        targets_curr[predicted_idx] = shifted_pt
        target_shifted = True

        # Landing: subject landed at target_orig (target 1's old position)
        landing_pos = target_pt
        # Distance to current shifted target
        dist_to_shifted = math.hypot(landing_pos[0] - shifted_pt[0], landing_pos[1] - shifted_pt[1])
        # Distance to all current targets
        dists = [math.hypot(landing_pos[0] - pt[0], landing_pos[1] - pt[1]) for pt in targets_curr]
        min_dist_dva = min(dists)

        is_hit = min_dist_dva < fovea_radius_dva
        points = 10 if is_hit else 0
        total_score += points
        if is_hit:
            hits += 1

        print(f"Trial {t}: Target {target_names[predicted_idx]} predicted -> SHIFTED by {shift_dist_dva:.1f} DVA | "
              f"Landing dist: {min_dist_dva:.2f} DVA | Hit: {is_hit} ({points} pts)")

        trials_data.append({
            "trial": t,
            "target_chosen": chosen_target_idx + 1,
            "target_predicted": predicted_idx + 1,
            "shifted": target_shifted,
            "landing_x_dva": landing_pos[0],
            "landing_y_dva": landing_pos[1],
            "min_distance_dva": min_dist_dva,
            "hit": is_hit,
            "points": points,
            "rt_ms": 185.0
        })

    acc_pct = (hits / num_trials) * 100.0
    results = {
        "score": total_score,
        "num_trials": num_trials,
        "hits": hits,
        "accuracy_pct": acc_pct,
        "trials": trials_data
    }
    print(f"\n[Headless Resultats] Score: {total_score} | Succes: {acc_pct:.1f}%")
    return results


def run_experiment(
    num_trials=12,
    source="auto",
    session="stream.eyelive",
    windowed=True,
    fovea_radius_dva=2.0,
    target_ecc_dva=7.0,
    shift_dist_dva=5.0,
    screen_width_cm=53.0,
    screen_dist_cm=60.0,
    test_mode=False
):
    """Run full PsychoPy experiment with visual display or fallback."""
    # Check PsychoPy availability
    try:
        from psychopy import visual, core, event, sound
        has_psychopy = True
    except ImportError:
        has_psychopy = False
        print("[PsychoPy] PsychoPy module not found in current Python environment.")
        print("[PsychoPy] Running headless simulation engine...")
        return run_headless_simulation(num_trials, fovea_radius_dva, target_ecc_dva, shift_dist_dva)

    if test_mode:
        print("[Game] Mode test actif.")

    # EyeLive Reader setup
    eyelive_reader = None
    if source in ("auto", "eyelive"):
        if os.path.exists(session) and StateReader is not None:
            try:
                eyelive_reader = StateReader(session)
                source = "eyelive"
                print(f"[EyeLive] Connecté au stream partagé: {session}")
            except Exception as e:
                print(f"[EyeLive] Avertissement: impossible d'ouvrir {session}: {e}")
                source = "mouse"
        else:
            source = "mouse"
    print(f"[Game] Mode de contrôle: {source.upper()}")

    # Setup PsychoPy Window
    win_size = (1024, 768) if windowed else (1920, 1080)
    win = visual.Window(
        size=win_size,
        fullscr=not windowed,
        units="pix",
        color=[-0.76, -0.76, -0.72],  # Dark slate gray
        allowGUI=windowed
    )

    ppd = compute_geometry(win.size[0], win.size[1], screen_width_cm, screen_dist_cm)
    fovea_radius_px = fovea_radius_dva * ppd
    target_ecc_px = target_ecc_dva * ppd
    target_radius_px = 0.8 * ppd
    shift_dist_px = shift_dist_dva * ppd

    # Stimuli definitions (Red, Green, Blue)
    colors_rgb = [
        [0.84, -0.49, -0.49],   # 1: Red
        [-0.57, 0.69, -0.41],   # 2: Green
        [-0.53, 0.10, 1.00]     # 3: Blue
    ]
    color_names = ["ROUGE", "VERT", "BLEU"]

    # Header texts
    header_trial = visual.TextStim(win, text="", pos=(-win.size[0]/2 + 100, win.size[1]/2 - 30), height=20, color="white")
    header_score = visual.TextStim(win, text="", pos=(win.size[0]/2 - 120, win.size[1]/2 - 30), height=20, color="gold")
    msg_center = visual.TextStim(win, text="", pos=(0, -60), height=22, color="white", wrapWidth=800)
    title_stim = visual.TextStim(win, text="JEU DU REGARD : ATTRAPEZ LE POINT !", pos=(0, 140), height=30, color="gold", bold=True)

    # Fixation cross
    fix_h = visual.Line(win, start=(-0.4 * ppd, 0), end=(0.4 * ppd, 0), lineColor="white", lineWidth=2)
    fix_v = visual.Line(win, start=(0, -0.4 * ppd), end=(0, 0.4 * ppd), lineColor="white", lineWidth=2)

    # Gaze cursor & Fovea circles
    gaze_cursor = visual.Circle(win, radius=6, fillColor="white", lineColor=None, opacity=0.7)
    fovea_ring = visual.Circle(win, radius=fovea_radius_px, fillColor=None, lineWidth=1.5, opacity=0.4)
    target_stims = [visual.Circle(win, radius=target_radius_px, fillColor=colors_rgb[i], lineColor=None) for i in range(3)]
    ghost_stims = [visual.Circle(win, radius=target_radius_px, fillColor=None, lineColor=colors_rgb[i], lineWidth=1.5, opacity=0.35) for i in range(3)]

    mouse = event.Mouse(win=win)

    def get_gaze_pos(t_stim_onset=0.0, targets_orig=None):
        """Retrieve current gaze coordinate in screen pixels relative to center (0, 0)."""
        if test_mode:
            if t_stim_onset == 0:
                return 0.0, 0.0, False, None, False
            elapsed = time.time() - t_stim_onset
            if elapsed < 0.120:
                return 0.0, 0.0, False, None, False
            else:
                # Saccade towards target 1 (Green)
                t2 = targets_orig[1]
                frac = min(1.0, (elapsed - 0.120) / 0.040)
                gx = frac * t2[0]
                gy = frac * t2[1]
                return gx, gy, (frac < 1.0), t2, True

        if source == "eyelive" and eyelive_reader is not None:
            state = eyelive_reader.latest()
            if state is not None:
                gx = state['x'] * ppd
                gy = state['y'] * ppd
                in_event = bool(state['in_event'])
                pred_valid = bool(state['pred_valid'])
                pred_pt = (state['pred_x'] * ppd, state['pred_y'] * ppd) if pred_valid else None
                return gx, gy, in_event, pred_pt, pred_valid

        # Mouse fallback
        mx, my = mouse.getPos()
        return mx, my, False, None, False

    # Instructions screen
    title_stim.draw()
    msg_center.text = (
        "Trois points colorés (Rouge, Vert, Bleu) vont apparaître.\n\n"
        "Regardez librement L'UN des trois points !\n\n"
        "L'algorithme tente en direct de deviner votre cible et de l'esquiver.\n"
        "Si votre fovéa atterrit à MOINS DE 2 DVA d'un point, vous gagnez 10 points !\n\n"
        "Appuyez sur ESPACE pour commencer (ESC pour quitter)..."
    )
    msg_center.draw()
    win.flip()

    if not test_mode:
        while True:
            keys = event.getKeys()
            if 'space' in keys:
                break
            if 'escape' in keys:
                win.close()
                return None
            core.wait(0.01)
    else:
        core.wait(0.05)

    # Main trial loop
    total_score = 0
    trials_log = []

    for t in range(1, num_trials + 1):
        # Generate equilateral triangle positions with randomized rotation
        base_angle = random.random() * 2 * math.pi
        angles = [base_angle + i * (2 * math.pi / 3) for i in range(3)]
        targets_orig = [(target_ecc_px * math.cos(a), target_ecc_px * math.sin(a)) for a in angles]
        targets_curr = list(targets_orig)

        # Phase A: Fixation Hold (400 ms)
        fixation_start = None
        while True:
            gx, gy, _, _, _ = get_gaze_pos(0.0, targets_orig)
            dist_center_dva = math.hypot(gx, gy) / ppd

            if dist_center_dva <= 1.5:
                if fixation_start is None:
                    fixation_start = time.time()
                elif (time.time() - fixation_start) >= 0.400:
                    break
            else:
                fixation_start = None

            header_trial.text = f"Essai {t} / {num_trials}"
            header_score.text = f"Score: {total_score} pts"
            header_trial.draw()
            header_score.draw()
            fix_h.draw()
            fix_v.draw()
            gaze_cursor.pos = (gx, gy)
            gaze_cursor.draw()
            win.flip()

            if 'escape' in event.getKeys():
                win.close()
                return None
            core.wait(0.005)

        # Phase B: Targets Onset & Real-Time Prediction Loop
        t_stim_onset = time.time()
        saccade_started = False
        t_saccade_start = 0.0
        target_shifted = False
        predicted_target = -1

        while (time.time() - t_stim_onset) < 1.5:
            gx, gy, in_event, pred_pt, pred_valid = get_gaze_pos(t_stim_onset, targets_orig)
            dist_fix_dva = math.hypot(gx, gy) / ppd

            if not saccade_started and (dist_fix_dva > 1.8 or in_event):
                saccade_started = True
                t_saccade_start = time.time()

            # Prediction & Dodge algorithm
            if saccade_started and not target_shifted:
                # 1. LandingNet prediction if available
                if pred_valid and pred_pt is not None:
                    dists = [math.hypot(targets_curr[i][0] - pred_pt[0], targets_curr[i][1] - pred_pt[1]) for i in range(3)]
                    predicted_target = dists.index(min(dists))
                else:
                    # 2. Kinematic trajectory vector cosine alignment
                    g_vec_len = math.hypot(gx, gy)
                    if g_vec_len > 1.0:
                        alignments = []
                        for i in range(3):
                            tx, ty = targets_orig[i]
                            t_len = math.hypot(tx, ty)
                            cos_sim = (gx * tx + gy * ty) / (g_vec_len * t_len)
                            alignments.append(cos_sim)
                        max_align = max(alignments)
                        if max_align > 0.50:
                            predicted_target = alignments.index(max_align)

                # If target predicted, apply dodge shift
                if predicted_target >= 0:
                    orig_x, orig_y = targets_orig[predicted_target]
                    orig_len = math.hypot(orig_x, orig_y)
                    out_x = (orig_x / orig_len) * shift_dist_px
                    out_y = (orig_y / orig_len) * shift_dist_px
                    targets_curr[predicted_target] = (orig_x + out_x, orig_y + out_y)
                    target_shifted = True
                    print(f"Trial {t}: Cible {color_names[predicted_target]} prédite -> DÉCALÉE de {shift_dist_dva:.1f} DVA!")

            # Saccade landing condition
            if saccade_started and (dist_fix_dva >= 4.0 or (time.time() - t_saccade_start) > 0.180):
                core.wait(0.100)
                gx, gy, _, _, _ = get_gaze_pos(t_stim_onset, targets_orig)
                break

            # Render stimuli
            header_trial.draw()
            header_score.draw()
            for i in range(3):
                target_stims[i].pos = targets_curr[i]
                target_stims[i].draw()
                if target_shifted and i == predicted_target:
                    ghost_stims[i].pos = targets_orig[i]
                    ghost_stims[i].draw()
            gaze_cursor.pos = (gx, gy)
            gaze_cursor.draw()
            win.flip()
            core.wait(0.002)

        # Phase C: Landing evaluation (< 2.0 DVA fovea radius)
        final_gaze = (gx, gy)
        dists_to_targets_dva = [math.hypot(targets_curr[i][0] - final_gaze[0], targets_curr[i][1] - final_gaze[1]) / ppd for i in range(3)]
        min_dist_dva = min(dists_to_targets_dva)
        chosen_idx = dists_to_targets_dva.index(min_dist_dva)

        is_hit = min_dist_dva < fovea_radius_dva
        points_won = 10 if is_hit else 0
        total_score += points_won

        feedback_color = [0.2, 0.9, 0.3] if is_hit else [0.9, 0.3, 0.3]
        if is_hit:
            verdict_text = "TOUCHÉ MALGRÉ L'ESQUIVE ! (+10 pts)" if (target_shifted and chosen_idx == predicted_target) else "TOUCHÉ ! FEINTE RÉUSSIE (+10 pts)"
        else:
            verdict_text = f"ESQUIVÉ ! L'algorithme a décalé {color_names[predicted_target]} (0 pt)" if target_shifted else "HORS CIBLE (0 pt)"

        trial_record = {
            "trial": t,
            "target_chosen": chosen_idx + 1,
            "target_predicted": (predicted_target + 1) if predicted_target >= 0 else 0,
            "shifted": target_shifted,
            "landing_x_dva": final_gaze[0] / ppd,
            "landing_y_dva": final_gaze[1] / ppd,
            "min_distance_dva": min_dist_dva,
            "hit": is_hit,
            "points": points_won,
            "rt_ms": (t_saccade_start - t_stim_onset) * 1000.0 if saccade_started else 0.0
        }
        trials_log.append(trial_record)

        # Feedback Screen (1 sec)
        t_fb = time.time()
        feedback_stim = visual.TextStim(win, text=verdict_text, pos=(0, -80), height=24, color=feedback_color)
        dist_stim = visual.TextStim(win, text=f"Distance la plus proche: {min_dist_dva:.2f} DVA (Seuil: < 2.0 DVA)", pos=(0, -120), height=18, color="gray")

        while (time.time() - t_fb) < 1.0:
            header_trial.draw()
            header_score.text = f"Score: {total_score} pts"
            header_score.draw()
            for i in range(3):
                target_stims[i].pos = targets_curr[i]
                target_stims[i].draw()
                fovea_ring.pos = targets_curr[i]
                fovea_ring.lineColor = colors_rgb[i]
                fovea_ring.draw()
            gaze_cursor.pos = final_gaze
            gaze_cursor.fillColor = feedback_color
            gaze_cursor.draw()
            feedback_stim.draw()
            dist_stim.draw()
            win.flip()
            core.wait(0.01)

        # Inter-trial interval (400 ms)
        win.flip()
        core.wait(0.400)

    # Final summary screen
    hits_total = sum(1 for r in trials_log if r["hit"])
    acc_pct = (hits_total / num_trials) * 100.0
    summary_msg = visual.TextStim(
        win,
        text=f"PARTIE TERMINÉE !\n\nScore Final : {total_score} points\n\nCibles atteintes (< 2 DVA) : {hits_total} / {num_trials} ({acc_pct:.1f}%)\n\nAppuyez sur une touche pour quitter...",
        pos=(0, 0),
        height=24,
        color="white"
    )
    summary_msg.draw()
    win.flip()

    if not test_mode:
        event.waitKeys()
    else:
        core.wait(0.05)
    win.close()

    results = {
        "score": total_score,
        "num_trials": num_trials,
        "hits": hits_total,
        "accuracy_pct": acc_pct,
        "trials": trials_log
    }

    out_file = f"results_psychopy_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Résultats] Score: {total_score} | Succès: {acc_pct:.1f}% | Sauvegardé: {out_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="PsychoPy Three-Target Gaze Prediction Game")
    parser.add_argument("--num-trials", type=int, default=12, help="Number of trials (default: 12)")
    parser.add_argument("--source", choices=["auto", "eyelive", "mouse"], default="auto", help="Gaze input source")
    parser.add_argument("--session", default="stream.eyelive", help="EyeLive shared memory file")
    parser.add_argument("--fullscreen", action="store_true", help="Launch in fullscreen mode")
    parser.add_argument("--fovea-radius-dva", type=float, default=2.0, help="Fovea scoring threshold in DVA")
    parser.add_argument("--target-ecc-dva", type=float, default=7.0, help="Target eccentricity in DVA")
    parser.add_argument("--shift-dist-dva", type=float, default=5.0, help="Dodge shift distance in DVA")
    parser.add_argument("--screen-width-cm", type=float, default=53.0, help="Screen width in cm")
    parser.add_argument("--screen-dist-cm", type=float, default=60.0, help="Eye to screen distance in cm")
    parser.add_argument("--test-mode", action="store_true", help="Run automated test simulation")
    args = parser.parse_args()

    run_experiment(
        num_trials=args.num_trials,
        source=args.source,
        session=args.session,
        windowed=not args.fullscreen,
        fovea_radius_dva=args.fovea_radius_dva,
        target_ecc_dva=args.target_ecc_dva,
        shift_dist_dva=args.shift_dist_dva,
        screen_width_cm=args.screen_width_cm,
        screen_dist_cm=args.screen_dist_cm,
        test_mode=args.test_mode
    )


if __name__ == "__main__":
    main()
