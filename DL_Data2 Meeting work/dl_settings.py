from pathlib import Path

DATASET_ROOT = Path(
    "/home/awais/Desktop/Awais Work 2026/KAIST Work/Team 1/"
    "KAIST Data/20260611_Data2"
)
PZT_FOLDER = "260609_pzt_network"
LDV_FOLDER = "2606010_Innocore_STS304_6PZT"
RESULTS_FOLDER = "paper5_dl_results"
RANDOM_SEED = 42
DEVICE = "auto"

FEATURE_FILE_CANDIDATES = [
    "physics_features_phase1_bpf.npz",
    "physics_features_phase1.npz",
    "physics_features_aug_bpf.npz",
    "physics_features_aug.npz",
    "physics_features.npz",
]

MAGNET_YX_MM = (215.0, 150.0)

PZT_POSITIONS_YX_MM = {
    1: (150.00, 275.00),
    2: (258.25, 212.50),
    3: (258.25, 87.50),
    4: (150.00, 25.00),
    5: (41.75, 87.50),
    6: (41.75, 212.50),
}
