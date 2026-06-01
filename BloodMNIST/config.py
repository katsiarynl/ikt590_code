CHANNELS = 3
RANDOM_SEED = 42
SAVE_DIR_SCORES = "BloodMNIST/results_BloodMNIST_scores"

labels= {
    0: "Basophil",
    1: "Eosinophil",
    2: "Erythroblast",
    3: "Immature granulocyte",
    4: "Lymphocyte",
    5: "Monocyte",
    6: "Neutrophil",
    7: "Platelet"
}

# class_id : augmentation_multiplier
AUGMENT_CLASSES = {
    0: 3,  # Basophil (strong minority)
    4: 3,  # Lymphocyte (strong minority)
    5: 3,  # Monocyte (strong minority)
    2: 2,  # Erythroblast (moderate minority)
    7: 2,  # Platelet (moderate minority)
}

N_TRIALS = 50 
N_EPOCHS_PER_TRIAL = 50
N_EPOCHS = 100
NUM_CLAUSES = 2000
DEVICE = "GPU"
DATA_DIR = "processed_data"


SAVE_MAPS_DIR = "saved_maps_raw"
SAVE_DIR_HEATMAPS = "saved_heatmaps"
ADAPTIVE_GAUSSIAN = "adaptive_gaussian"
ADAPTIVE_MEAN = "adaptive_mean"
OTSU = "otsu"
CANNY = "canny"
COLOR_ENCODING = "color_encoding"
LOCAL_HEATMAP = "lh"
GLOBAL_HEATMAP = "gh"
BLOODMNIST = "blood"
AGGREGATED_HEATMAP = "aggr"
CHANNEL_HEATMAP = "chan"
RES_HEATMAP_R = "res_r"
RES_HEATMAP_G = "res_g"
RES_HEATMAP_B = "res_b"
RES_HEATMAP = "res"
CHANNEL_LEVEL_HEATMAP = "chan_level"
TITLE_GH_POS = "Positive clauses"
TITLE_GH_NEG = "Negative clauses"
