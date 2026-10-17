# Manuel d'utilisation d'EyeLive

EyeLive enregistre en temps réel le flux d'un oculomètre, détecte automatiquement les saccades (et les clignements), vous laisse corriger les étiquettes en quelques minutes, apprend de vos corrections et peut déclencher votre expérience sur des événements. C'est un petit programme (une fenêtre, et son jumeau en ligne de commande `eyelive_cli`), sans installateur et sans droits d'administrateur. (English version: [MANUAL.md](MANUAL.md).)

## 1. Installation
Décompressez la version dans un dossier qui vous appartient et suivez [INSTALL_UNSIGNED.md](INSTALL_UNSIGNED.md) (aussi dans le zip sous `INSTALL.txt`) : vérifiez le SHA-256, débloquez le zip sous Windows, « Ouvrir quand même » sous macOS, ajoutez le dossier aux exclusions de l'antivirus si besoin. Le zip contient `eyelive` (fenêtre), `eyelive_cli` (sans fenêtre) et `models/` (réseaux pré-entraînés : gardez ce dossier à côté du programme).

## 2. Ce que fait le programme de vos données
* Un **fichier de session** (`.eyelive`) est créé au clic sur Start, à la taille que vous annoncez (pré-alloué : il ne peut pas remplir le disque pendant l'expérience). Il contient les échantillons, les événements détectés, vos corrections et un petit **bloc d'état** temps réel que MATLAB/Psychtoolbox ou Python/PsychoPy peuvent lire pendant l'enregistrement (section 9).
* Rien ne quitte l'ordinateur. Le port UDP n'est ouvert que pendant une session.
* Trois classes par défaut : **fixation** (la poursuite lente y reste), **saccade** (les oscillations post-saccadiques y restent), **clignement**. Jusqu'à cinq classes sont possibles.

## 3. Première session en quatre clics : l'assistant et le bouton Start
Au premier lancement, une fenêtre pose quatre questions (votre oculomètre, le signal, ce que vous voulez, un nom) et enregistre les réponses dans un **profil**. Ensuite, enregistrer = choisir le profil et appuyer sur le grand bouton vert **Start recording**.
* **Les quatre modes** en haut regroupent les onglets : *Record* (Setup, Live, Triggers), *Review* (Review, Detection, Train), *Analyze* (Predict, Export, Log), *Help*. *Advanced* montre tous les onglets, dont l'ancien onglet *Acquire* (section 3b).
* **Le voyant de santé** (en haut à droite) n'est jamais la seule information : il écrit *idle*, *OK*, *CHECK* ou *PROBLEM* et la raison (aucun échantillon depuis 2 s, paquets perdus, latence de détection élevée, fichier presque plein).
* **Les profils** (`~/.eyelive/profiles/*.profile`, un fichier texte copiable sur un autre ordinateur) contiennent l'oculomètre, le port, la fréquence, les unités et l'écran, le détecteur, le réseau d'atterrissage (section 9), les règles de déclenchement et la durée prévue. L'onglet *Setup* les modifie, les enregistre et lance les **vérifications** (port libre, espace disque, fichier de règles lisible, échelle).
  Chaque session est enregistrée sous `~/EyeLive/<profil>_<date>_<heure>.eyelive`, jamais par-dessus une autre. Si le profil a un fichier de règles de déclenchement, une erreur dedans empêche le *démarrage*, pas l'expérience.
* **À côté de chaque session** : `<session>.provenance.json` (version du programme, détecteur, profil, réseau d'atterrissage, SHA-256 des données et des fichiers de modèle : de quoi reproduire une analyse) et `<session>.report.html` (une page : durée, échantillons manquants, trous, taux de saccades, amplitudes, séquence principale, avertissements du type « le flux est probablement en pixels »).
* **Onglet Export** : CSV des événements, TSV + JSON de type BIDS, fichier MAT (`t, x, y, valid, label, expert, events` ; MATLAB, Octave, `scipy.io.loadmat`), ou le rapport.
* **Onglet Help** : les vérifications, l'emplacement des fichiers, les problèmes usuels, et un texte de diagnostic (version, profil, vérifications, statistiques, aucune donnée de regard) à coller dans un message au support.

## 3b. L'onglet Acquire (mode Advanced)
![Onglet Acquire](img/tab_Acquire.png)
1. **Source.** *Simulator* (aucun oculomètre nécessaire : essayez tout avec), *File replay* (un CSV `t,x,y[,valid]`) ou *UDP stream* (l'ordinateur de l'oculomètre envoie les échantillons à un port ; section 3.1).
2. **Fréquence et durée.** Laissez la fréquence à 0 : EyeLive la mesure sur les premières 500 ms et l'arrondit à la fréquence standard la plus proche (60, 120, 250, 300, 500, 1000, 1200, 2000 Hz). La durée prévue fixe la pré-allocation, affichée pendant la saisie avec l'espace disque libre.
3. **Unités du flux.** *Degrees* ; *pixels avec pixels par degré connus* ; *pixels avec la géométrie de l'écran* (largeur en cm, distance en cm, résolution horizontale : le programme calcule les pixels par degré) ; ou *find it automatically* (section 8).
4. **Fichier de session** et **détecteur au démarrage** (`hmm_accel`, sans étiquettes, par défaut). Cliquez **Start** ; une erreur (port déjà utilisé, dossier absent, espace insuffisant) s'écrit à côté du bouton. **Stop** ferme proprement le fichier.

### 3.1 Envoyer les échantillons (protocole v1)
Datagrammes UDP vers le port, petit-boutiste. Binaire : `uint32 0x31455945 ("EYE1")`, `uint32 n`, puis `n` enregistrements `double t_secondes, float x, float y, uint32 flags` (20 octets ; flags 1 = valide, 2 = clignement). Texte : un échantillon par ligne, `t x y [valid]`. De 1 à 50 échantillons par datagramme. `cpp/app/tools/send_udp.py` rejoue un CSV dans ce format ; `cpp/app/tools/pylink_sender_template.py` est un modèle pour relayer les échantillons EyeLink (modèle non testé).

## 4. Suivre le flux : onglet Live
![Onglet Live](img/tab_Live.png)
Position x et y (bleu, orange), vitesse (vert) et événements en bandes colorées ; **les étiquettes viennent du détecteur que vous avez choisi** : `hmm_accel` et les réseaux affichent chaque événement quand il est final (environ 115 ms après le mouvement, avec début, fin et amplitude exacts), `fast_threshold` affiche une bande 3 à 6 ms après le début dont la fin suit le dernier échantillon pendant la saccade (marquée `*` dans la liste jusqu'à l'événement final). Le détecteur rapide tourne toujours à côté du détecteur choisi, mais seulement pour piloter les déclencheurs ; la carte du regard montre les 500 dernières ms. Molette = zoom, glisser = déplacer, double clic = aller à la fin, *follow* garde les données récentes visibles. Le tableau de droite liste les derniers événements. La barre d'état donne la fréquence mesurée, la mémoire, le CPU et la latence de détection.

## 5. Détecteurs : onglet Detection
![Onglet Detection](img/tab_Detection.png)
`hmm_accel` (sans étiquettes, bornes affinées par le zéro d'accélération : à utiliser en premier), `fast_threshold` (le plus rapide : un événement 3 à 6 ms après le début, pour les déclencheurs), les réseaux U'n'Eye pré-entraînés, les TCN causal et bidirectionnel, et les modèles que vous entraînez vous-même (`user:<nom>`). **Use now** change de détecteur pendant l'enregistrement sans perdre ni échantillon ni événement. **Fit the convention to my corrections** apprend comment vous placez débuts et fins et quelles amplitudes vous gardez.

## 6. Corriger les étiquettes : onglet Review
![Onglet Review](img/tab_Review.png)
1. Choisissez le **pré-étiqueteur** et cliquez *Pre-label all* (ou *window*) : les événements du détecteur apparaissent en bandes et dans la liste.
2. Corrigez : **cliquez** un événement (sur le tracé ou dans la liste) pour le sélectionner : un contour blanc apparaît et une ligne de boutons indique son instant, sa durée, son amplitude et sa classe. **Delete selected** (ou la touche Suppr / Retour arrière) le supprime aussitôt. **Glissez ses bords** pour déplacer le début ou la fin. Pour **ajouter** un événement, cliquez **Add event** (ou **A**), puis **cliquez son début et cliquez sa fin** (ou glissez sur le tracé) ; **Échap** quitte le mode ajout. Les touches **1-4** ou les boutons de classe changent la classe, **S** coupe au curseur, **M** fusionne avec le suivant, **Z / Y** annuler / rétablir. Molette = zoom, défilement horizontal ou glisser = déplacer.
3. Appuyez sur **R** (ou *Mark window reviewed*) pour chaque portion vérifiée ; **X** exclut une portion de l'entraînement. *Next uncertain* (**]**) saute aux événements dont le détecteur était le moins sûr.
4. La barre compare les secondes revues à la cible de 50 s ; la ligne d'accord compare le détecteur à vos corrections (F1 par événement et kappa).

### 6.1 Annotation assistée (désactivée par défaut : cochez *Assisted annotation* en haut de l'onglet Review)
Inutile d'attendre la fin de l'enregistrement. Le détecteur sans étiquettes étiquette les données à mesure qu'elles arrivent (blocs de 2 s) : vous pouvez corriger dès les premières secondes. Chaque correction rend son voisinage (±0,5 s) « étiqueté par vous ». **5 s après votre dernière correction**, un U'n'Eye est entraîné en arrière-plan (un cœur pendant l'enregistrement) sur vos étiquettes là où vous avez revu, et sur les étiquettes automatiques ailleurs. La barre en haut de l'onglet montre l'époque par rapport aux ~50 habituelles ; l'entraînement continue tant que la perte de validation diminue et s'arrête par arrêt précoce. Quand le modèle est assez bon (F1 de validation d'au moins 0,6, pour qu'un modèle qui ne prédit rien ne remplace jamais les étiquettes), il étiquette les fenêtres qui suivent celles que vous avez revues, et les nouvelles données, à la place du détecteur sans étiquettes ; un modèle rejeté continue d'apprendre de vos corrections suivantes. Après vos corrections suivantes, un nouveau modèle est entraîné. Ouvrez *Quality of each labeler* : pour chaque étiqueteur (détecteur sans étiquettes, modèle 1, 2…), l'accord de SES étiquettes avec vos étiquettes finales sur les fenêtres qu'il a étiquetées puis que vous avez revues. Quand le dernier modèle est aussi bon que nécessaire sur des fenêtres que vous n'aviez pas encore touchées, vous pouvez arrêter d'annoter.

## 7. Apprendre de vos corrections : onglet Train
![Onglet Train](img/tab_Train.png)
Par défaut : 3 classes, les 50 dernières secondes revues, rotations de la trace activées (4 copies ; à désactiver si la tâche dépend fortement de la direction), départ de poids aléatoires (ou d'un réseau pré-entraîné, utile sur peu de données). **Train** tourne en arrière-plan (un cœur à basse priorité pendant l'enregistrement, tous les cœurs ensuite ; environ 5 s par époque et par cœur pour 50 s de données). Les courbes montrent les pertes d'entraînement et de validation et le F1 par événement. **Deploy as detector** enregistre le modèle près de la session et ajoute `user:<nom>` aux détecteurs. Vous pouvez augmenter la quantité de données et relancer.

## 8. Unités et échelle
Les réseaux d'EyeLive travaillent en degrés. Si le flux est en pixels, donnez les pixels par degré (le mieux) ou laissez EyeLive les estimer : il compare le réseau à de nombreuses échelles avec le détecteur indépendant des unités et garde l'échelle où ils s'accordent (l'onglet Log l'affiche ; un avertissement signale une confiance faible, saisissez alors la valeur). Une échelle approximative suffit à la détection ; les amplitudes en degrés ne valent que ce que vaut l'échelle.

## 9. Déclencheurs et sortie temps réel : onglets Triggers et Predict
![Onglet Triggers](img/tab_Triggers.png)
Une règle se déclenche sur un événement d'une classe (ou toutes), éventuellement dans une plage d'amplitude, **après un délai**, et envoie un message UDP (`udp:127.0.0.1:6000`), écrit une ligne de journal (`log:<fichier>`) ou affiche (`stdout`). Ancres : *onset (provisional)* se déclenche 3 à 6 ms après le début du mouvement (l'amplitude n'est pas encore connue : non combinable avec une plage d'amplitude), *onset (final)* et *offset* utilisent l'événement final. Variables de la charge : `{t}` temps de l'événement, `{amp}` amplitude en degrés, `{cls}` classe. L'onglet Log liste les instants demandés et réels.

![Onglet Predict](img/tab_Predict.png)
Aucun matériel TTL : votre programme de stimulation lit le **bloc d'état** du fichier de session (dernier regard, vitesse, compteurs d'événements, dernier événement). MATLAB / Psychtoolbox : `m = memmapfile(session, 'Offset', 256, 'Format', 'uint8', 'Repeat', 256);` puis `eyelive_state(m.Data)` (`cpp/app/readers/matlab/`). Python / PsychoPy : `from eyelive_reader import StateReader` (`cpp/app/readers/python/`). L'onglet **Predict** (et la case *forecast +20 ms* de l'onglet Live, désactivée par défaut) trace la prévision de la position de l'œil 10 et 20 ms à l'avance par-dessus la mesure (x en magenta, y en jaune, décalés au temps futur et dépassant de 20 ms le dernier échantillon, avec leur intervalle à 80 %) et mesure son erreur sur les 10 dernières secondes, comparée à « l'œil reste où il est ». Il faut un flux à 1 kHz ; la dernière prévision est aussi dans le bloc d'état.

## 10. Après l'enregistrement, sans fenêtre : eyelive_tool
```sh
eyelive_tool info session.eyelive                                  # le rapport en texte
eyelive_tool export session.eyelive --csv events.csv --bids sub-01_task-x --mat session.mat --report report.html
eyelive_tool redetect sessions/ --detector fast_threshold --out redo/ [--recursive]     # ré-étiqueter un dossier avec un autre détecteur ou modèle ; les sessions ne sont pas modifiées
eyelive_tool check --profile "Lab A" --session-dir ~/EyeLive       # port, fichiers, espace disque, échelle
eyelive_tool profiles list | show NOM | new NOM --from "Tracker over UDP, screen pixels"
```
Code de sortie 0 si tout va bien, 1 en cas d'échec (message sur stderr), 2 pour une erreur d'usage.

## 10b. Sans fenêtre : eyelive_cli
```sh
eyelive_cli --sim --duration 5 --session demo.eyelive
eyelive_cli --udp 5005 --duration 3600 --session sujet1.eyelive --scale auto
eyelive_cli --sim --duration 20 --prelabel-seconds 15 --train --session train.eyelive
```
Voir `CLI.txt` (cpp/app/cli/README.md) : rejeu, fichier de règles, CSV d'événements, codes de sortie. `docs/eyelive/check_commands.sh` exécute les commandes de ce manuel.

## 11. Dépannage
| symptôme | que faire |
|---|---|
| Windows ou macOS refuse de lancer le programme | [INSTALL_UNSIGNED.md](INSTALL_UNSIGNED.md) : débloquer, Ouvrir quand même, exclusion antivirus |
| le pare-feu Windows demande l'accès réseau | choisir *Réseaux privés* ; inutile quand l'ordinateur de l'oculomètre est le même (127.0.0.1) |
| Start échoue : « cannot create ... : File exists » | un fichier de session n'est jamais écrasé : choisissez un autre nom |
| Start échoue : « no samples received to measure the sampling rate » | rien n'arrive sur le port : vérifiez l'adresse, le port et l'émetteur ; déclarez la fréquence pour éviter la mesure |
| paquets perdus / trous dans la barre d'état | l'émetteur est plus rapide que le réseau ou le PC : envoyez 5 à 20 échantillons par datagramme, fermez les programmes lourds |
| « automatic unit scale has low confidence » | saisissez les pixels par degré ou la géométrie de l'écran dans l'onglet Acquire |
| rien n'est détecté | vérifiez les unités (un flux en pixels lu comme des degrés est trop petit) ; essayez un autre détecteur ; regardez l'onglet Live |
| la fenêtre ne s'ouvre pas (Linux) | OpenGL 3.2 est requis : `glxinfo | grep "OpenGL version"` |
| la session est énorme | le fichier est pré-alloué pour la durée prévue : réduisez-la ; c'est un fichier binaire simple que vous pouvez copier |
