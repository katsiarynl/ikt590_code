CHANNELS = 1

DATA_DIR = "processed_data"
RANDOM_SEED = 42
BETA = 2.0
N_TRIALS = 50 
N_EPOCHS_PER_TRIAL = 50
N_EPOCHS = 100
NUM_CLAUSES = 2000
labels =  {
    0: "Malignant",  # Cancerous tumor
    1: "Benign"      # Non-cancerous tumor
}
SAVE_DIR_SCORES = "BreastMNIST/results_BreastMNIST_scores"
DEVICE = "GPU"
MODEL_DIR = "saved_models"

SAVE_MAPS_DIR = "saved_maps_raw"
SAVE_DIR_HEATMAPS = "saved_heatmaps"
ADAPTIVE_GAUSSIAN = "adaptive_gaussian"
ADAPTIVE_MEAN = "adaptive_mean"
OTSU = "otsu"
CANNY = "canny"
COLOR_ENCODING = "color_encoding"
LOCAL_HEATMAP = "lh"
GLOBAL_HEATMAP = "gh"
BREASTMNIST = "breast"
AGGREGATED_HEATMAP = "aggr"
CHANNEL_HEATMAP = "chan"
CHANNEL_LEVEL_HEATMAP = "chan_level"
TITLE_GH_POS = "Positive clauses"
TITLE_GH_NEG = "Negative clauses"
RES_HEATMAP = "res"
