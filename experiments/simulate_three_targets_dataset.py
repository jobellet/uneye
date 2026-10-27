#!/usr/bin/env python3
"""
Real-Time Gaze Prediction & Dodge Game Simulation on Real Eye-Tracking Datasets
================================================================================
Evaluates the three-target prediction and dodge algorithm using real large saccades
from U'n'Eye benchmark datasets (dataset1 & dataset2: human recordings up to 12.5°).

Methodology:
1. Extract large saccades (amplitude >= min_amp DVA) with pre-saccadic fixation
   and post-saccadic landing coordinates.
2. For each trial:
   - Center coordinates relative to pre-saccadic fixation (0, 0).
   - "Place one of the 3 targets at the exact post-saccadic landing position":
     Target k (Red, Green, or Blue) is placed at P_end = (x_end, y_end).
     The other two targets are placed at the same eccentricity at theta ± 120°.
3. Replay the continuous trajectory sample by sample (1000 Hz):
   - Detect saccade initiation (displacement >= 1.5 DVA).
   - Compute angular cosine alignment between early saccade vector and target candidates.
   - Predict candidate target with highest alignment.
   - Dodge: shift the predicted target outward (+shift_dist DVA).
4. Saccade landing evaluation:
   - Check if landing fovea is within < fovea_radius DVA (2.0 DVA) of ANY target.
   - Award points (+10 pts) and log metrics:
     * Prediction Accuracy (did it pick the target the eye aimed at?)
     * Trigger Latency (ms after onset)
     * Saccade Progress (% of amplitude at trigger)
     * Dodge Success Rate (was the target successfully kept > 2.0 DVA away?)
"""

import os
import sys
import glob
import json
import math
import argparse
from datetime import datetime
import numpy as np


def find_data_dir():
    """Find data directory relative to repository root."""
    candidate_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"),
        os.path.join(os.getcwd(), "data"),
    ]
    for p in candidate_paths:
        if os.path.isdir(p):
            return os.path.abspath(p)
    raise FileNotFoundError("Impossible de trouver le dossier data/ de U'n'Eye.")


def extract_large_saccades(data_dir, datasets=("dataset1", "dataset2"), min_amp=3.0):
    """
    Extract large saccades (>= min_amp DVA) from U'n'Eye benchmark CSV files.
    Returns list of dicts with continuous trajectories and ground-truth timing.
    """
    saccades = []
    for ds in datasets:
        for s in ("setA", "setB"):
            xf = glob.glob(os.path.join(data_dir, ds, f"*_X_{s}.csv"))
            yf = glob.glob(os.path.join(data_dir, ds, f"*_Y_{s}.csv"))
            lf = glob.glob(os.path.join(data_dir, ds, f"*_Labels_{s}.csv"))
            if not (xf and yf and lf):
                continue

            X = np.loadtxt(xf[0], delimiter=",")
            Y = np.loadtxt(yf[0], delimiter=",")
            L = np.loadtxt(lf[0], delimiter=",")

            n_trials = min(len(X), len(Y), len(L))
            for r in range(n_trials):
                row_l, row_x, row_y = L[r], X[r], Y[r]
                in_s, s_idx = False, 0
                for i in range(len(row_l)):
                    if row_l[i] == 1 and not in_s:
                        in_s, s_idx = True, i
                    elif row_l[i] == 0 and in_s:
                        in_s = False
                        e_idx = i - 1
                        dx = row_x[e_idx] - row_x[s_idx]
                        dy = row_y[e_idx] - row_y[s_idx]
                        amp = float(np.hypot(dx, dy))

                        if amp >= min_amp:
                            # Pre-saccade window (100 ms) and post-saccade window (150 ms)
                            t_pre = max(0, s_idx - 100)
                            t_post = min(len(row_x), e_idx + 150)
                            saccades.append({
                                "dataset": ds,
                                "split": s,
                                "row": r,
                                "s_idx": s_idx,
                                "e_idx": e_idx,
                                "amp": amp,
                                "duration_ms": e_idx - s_idx + 1,
                                "x": row_x[t_pre:t_post],
                                "y": row_y[t_pre:t_post],
                                "onset": s_idx - t_pre,
                                "offset": e_idx - t_pre,
                            })
    return saccades


def simulate_game_on_saccades(saccades, shift_dist=5.0, fovea_radius=2.0, trigger_dist=1.5):
    """
    Run the three-target game simulation on extracted saccade trajectories.
    """
    results_trials = []
    latencies = []
    frac_amplitudes = []
    landing_distances = []
    correct_predictions = 0
    shifted_count = 0
    subject_hits = 0

    target_names = ["ROUGE", "VERT", "BLEU"]

    for idx, s in enumerate(saccades):
        x, y = s["x"], s["y"]
        onset, offset = s["onset"], s["offset"]

        # Pre-saccadic fixation baseline
        w_pre = max(0, onset - 40)
        x0 = float(np.mean(x[w_pre:onset])) if onset > 0 else float(x[0])
        y0 = float(np.mean(y[w_pre:onset])) if onset > 0 else float(y[0])

        # Post-saccadic landing position
        w_post = min(len(x), offset + 40)
        x_end = float(np.mean(x[offset:w_post])) if offset < len(x) else float(x[-1])
        y_end = float(np.mean(y[offset:w_post])) if offset < len(y) else float(y[-1])

        # Centered trajectory
        xc = x - x0
        yc = y - y0
        p_final = np.array([x_end - x0, y_end - y0])
        r_final = float(np.linalg.norm(p_final))
        theta_final = float(np.arctan2(p_final[1], p_final[0]))

        # Place target k at the exact final landing position
        chosen_k = idx % 3  # Rotate chosen target index (0=Red, 1=Green, 2=Blue)
        targets_orig = np.zeros((3, 2))
        for i in range(3):
            ang = theta_final + (i - chosen_k) * (2.0 * math.pi / 3.0)
            targets_orig[i] = [r_final * math.cos(ang), r_final * math.sin(ang)]
        targets_curr = targets_orig.copy()

        # Replay trajectory sample-by-sample (1 kHz)
        predicted_idx = -1
        pred_latency = 0
        pred_frac = 0.0

        for t in range(onset, offset + 1):
            gx, gy = xc[t], yc[t]
            dist_c = math.hypot(gx, gy)

            # Trigger condition: eye moves away from fixation
            if dist_c >= trigger_dist and predicted_idx == -1:
                vec = np.array([gx, gy])
                cos_sims = [
                    np.dot(vec, targets_orig[i]) / (np.linalg.norm(vec) * np.linalg.norm(targets_orig[i]))
                    for i in range(3)
                ]
                best_i = int(np.argmax(cos_sims))
                if cos_sims[best_i] > 0.50:
                    predicted_idx = best_i
                    pred_latency = t - onset
                    pred_frac = dist_c / max(0.1, r_final)

                    # Dodge: shift predicted target outward by shift_dist
                    out_dir = targets_orig[predicted_idx] / np.linalg.norm(targets_orig[predicted_idx])
                    targets_curr[predicted_idx] = targets_orig[predicted_idx] + out_dir * shift_dist
                    break

        if predicted_idx != -1:
            shifted_count += 1
            latencies.append(pred_latency)
            frac_amplitudes.append(pred_frac)
            if predicted_idx == chosen_k:
                correct_predictions += 1

        # Landing scoring: distance to all current targets
        dists = [float(np.linalg.norm(p_final - targets_curr[i])) for i in range(3)]
        min_dist = min(dists)
        closest_target = int(np.argmin(dists))
        landing_distances.append(min_dist)

        is_hit = min_dist < fovea_radius
        if is_hit:
            subject_hits += 1

        results_trials.append({
            "trial": idx + 1,
            "dataset": s["dataset"],
            "amplitude_deg": s["amp"],
            "duration_ms": s["duration_ms"],
            "target_intended": chosen_k + 1,
            "target_predicted": (predicted_idx + 1) if predicted_idx != -1 else 0,
            "prediction_correct": (predicted_idx == chosen_k),
            "prediction_latency_ms": pred_latency,
            "saccade_frac_at_trigger": pred_frac,
            "shifted": (predicted_idx != -1),
            "min_landing_dist_dva": min_dist,
            "closest_target": closest_target + 1,
            "hit": is_hit,
            "points": 10 if is_hit else 0
        })

    n_tot = len(saccades)
    acc_pct = (correct_predictions / max(1, shifted_count)) * 100.0
    shift_rate = (shifted_count / max(1, n_tot)) * 100.0
    dodge_rate = ((n_tot - subject_hits) / max(1, n_tot)) * 100.0
    hit_rate = (subject_hits / max(1, n_tot)) * 100.0

    summary = {
        "num_saccades": n_tot,
        "shift_dist_dva": shift_dist,
        "fovea_radius_dva": fovea_radius,
        "trigger_dist_dva": trigger_dist,
        "shift_triggered_count": shifted_count,
        "shift_triggered_pct": shift_rate,
        "prediction_correct_count": correct_predictions,
        "prediction_accuracy_pct": acc_pct,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "median_latency_ms": float(np.median(latencies)) if latencies else 0.0,
        "mean_saccade_progress_pct": float(np.mean(frac_amplitudes) * 100.0) if frac_amplitudes else 0.0,
        "mean_landing_distance_dva": float(np.mean(landing_distances)) if landing_distances else 0.0,
        "dodge_success_pct": dodge_rate,
        "subject_hits_count": subject_hits,
        "subject_hit_rate_pct": hit_rate,
        "total_score": subject_hits * 10,
        "trials": results_trials
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Three-Target Gaze Prediction Game Simulation on Real Saccades")
    parser.add_argument("--min-amp", type=float, default=3.0, help="Minimum saccade amplitude in degrees (default: 3.0)")
    parser.add_argument("--shift-dist", type=float, default=5.0, help="Dodge shift distance in degrees (default: 5.0)")
    parser.add_argument("--fovea-radius", type=float, default=2.0, help="Fovea scoring threshold in degrees (default: 2.0)")
    parser.add_argument("--trigger-dist", type=float, default=1.5, help="Saccade displacement trigger in degrees (default: 1.5)")
    parser.add_argument("--output", default="results_simulation_dataset.json", help="Output JSON path")
    args = parser.parse_args()

    data_dir = find_data_dir()
    print(f"Chargement des saccades depuis : {data_dir} (amplitude >= {args.min_amp:.1f}°)...")
    saccades = extract_large_saccades(data_dir, datasets=("dataset1", "dataset2"), min_amp=args.min_amp)
    print(f"-> {len(saccades)} saccades extraites (max amp: {max(s['amp'] for s in saccades):.2f}°).")

    print("\nLancement de la simulation du jeu (cible placée à la position finale)...")
    res = simulate_game_on_saccades(
        saccades,
        shift_dist=args.shift_dist,
        fovea_radius=args.fovea_radius,
        trigger_dist=args.trigger_dist
    )

    print("\n" + "=" * 70)
    print(" RÉSULTATS DE LA SIMULATION DU JEU SUR SACCADES RÉELLES")
    print("=" * 70)
    print(f"Saccades testées (amp >= {args.min_amp:.1f}°) : {res['num_saccades']}")
    print(f"Détection de saccade déclenchée       : {res['shift_triggered_count']} / {res['num_saccades']} ({res['shift_triggered_pct']:.1f}%)")
    print(f"Précision de prédiction de la cible   : {res['prediction_correct_count']} / {res['shift_triggered_count']} ({res['prediction_accuracy_pct']:.1f}%)")
    print(f"Latence moyenne de prédiction         : {res['mean_latency_ms']:.1f} ms après onset (médiane: {res['median_latency_ms']:.1f} ms)")
    print(f"Progression de la saccade au trigger  : {res['mean_saccade_progress_pct']:.1f}% de l'amplitude totale")
    print(f"Distance moyenne d'atterrissage       : {res['mean_landing_distance_dva']:.2f} DVA (seuil: < {args.fovea_radius:.1f} DVA)")
    print(f"Taux d'esquive réussie (Dodge Rate)   : {res['dodge_success_pct']:.1f}%")
    print(f"Points marqués par le sujet (< 2 DVA) : {res['subject_hits_count']} / {res['num_saccades']} ({res['subject_hit_rate_pct']:.1f}%) | Score: {res['total_score']} pts")
    print("=" * 70)

    # Save output JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nRapport détaillé sauvegardé dans : {args.output}")


if __name__ == "__main__":
    main()
