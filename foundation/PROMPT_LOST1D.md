# Prompt pour l'agent qui exécute le code sur la machine de l'owner (à coller tel quel)

---

Nous venons d'ajouter sur la branche `vibe/lost-ncut-memory-readouts-d0df29` (dépôt déjà fusionné/tiré dans ton working tree) un seul nouveau fichier de code et un fichier de références :

- `foundation/lost1d.py` — nouveaux readouts sans label qui n'utilisent PAS l'attention [CLS] du DINO 1D (qui était anti-corrélée aux saccades). Pour chaque fenêtre glissante (même géométrie que `dino1d.attention_maps` : fenêtres de 480 échantillons, crop central de 336, stride 84), on prend les keys et les tokens du DERNIER bloc du ViT de `foundation/dino1d.py`, et on calcule :
  1. **LOST** (`lost`) : graine = token de degré minimal du graphe de similarité cosinus des keys ; carte = similarité de chaque token à la graine.
  2. **TokenCut** (`tokencut`) : vecteur de Fiedler (2e plus petit vecteur propre du Laplacien normalisé) du même graphe.
  3. **MaskDistill** (`kmeans`) : k-means (k=2) sur les tokens L2-normalisés de la fenêtre ; carte positive pour le cluster minoritaire.
  4. **PatchCore-like** (`memory`) : mémoire de 16 384 tokens construite sur le split TRAIN NON ÉTIQUETÉ du dataset testé (jamais de label lu) ; carte = distance cosinus au plus proche voisin de la mémoire.
  Baselines dans le même rapport : le canal vitesse détendue (`speed_channel`, le plancher à battre) et le réseau NON-ENTRAÎNÉ (contrôle, comme dans `dino_eval.py`).
- `foundation/REFERENCES.md` — URL des papiers qui inspirent chaque méthode (LOST, TokenCut, MaskDistill, PatchCore, CutLER, I-JEPA, AnomalyBERT, Registers…).

## Ce que tu dois faire

1. Vérifier que `foundation/runs/dino1d.pt` existe (l'entraînement DINO déjà fait). Si oui :
   ```
   python foundation/lost1d.py
   ```
   (CPU/GPU selon `foundation/train_lodo.py::DEV` ; ~60 essais par dataset, 20 pour Andersson ; sortie dans `foundation/runs/lost1d.json`, format : une ligne par dataset, `trained` et `untrained`, AUC par readout.)
   Si le checkpoint n'existe pas, le script continue tout seul en contrôle-untrained seulement (il affiche un message).
2. Ne PAS réentraîner le DINO pour l'instant ; on évalue le checkpoint existant.
3. Aucun autre fichier n'a été modifié : ne touche ni à `dino1d.py` ni à `dino_eval.py`. Si tu veux ajouter quelque chose, arrête-toi et demande.
4. Le script a été testé de bout en bout sur d1–d3 en CPU (4 essais) ; Andersson n'a pas pu être testé ici car `data/Andersson/` n'est pas dans la sandbox — sur cette machine il doit tourner. Si Andersson plante, envoie la trace exacte.

## Comment lire les résultats (règle de décision, ne rien sélectionner sur le test au-delà de ce qui est déjà la convention du repo)

- Ce qui compte : la colonne `memory` du réseau ENTRAÎNÉ vs `speed_channel`. En contrôle untrained, `memory` faisait déjà 0.63–0.66 (vs 0.80–0.90 pour la vitesse) — si le ViT entraîné la fait monter au-dessus de la vitesse sur un dataset, c'est le signal que les tokens contiennent de l'information saccade au-delà du trivial, et on passe à l'étape suivante (seuil + min-duration + event F1 via `foundation/compare.py::event_f1`, puis distillation HMM → ViT à la CutLER).
- `lost` / `tokencut` / `kmeans` : en untrained ils sont ≈ chance (0.45–0.63). S'ils restent à la chance en trained, ça confirme que le graphe de tokens du ViT actuel ne sépare pas saccade/fixation, et il faudra soit des registres (Darcet et al. 2024, cf. REFERENCES.md), soit un prétexte I-JEPA à l'échelle d'une saccade.
- Note tout dans un commentaire du style des rapports existants (pas de number tapé à la main : le JSON `lost1d.json` fait foi).
