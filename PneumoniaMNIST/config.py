CHANNELS = 1
RANDOM_SEED = 42
DATA_DIR = "processed_data"
BETA = 2.0
labels = {
    0: "Normal", 
    1: "Pneumonia"      
}
N_TRIALS = 50 
N_EPOCHS_PER_TRIAL = 50
N_EPOCHS = 100
NUM_CLAUSES = 2000

DEVICE = "GPU"
MODEL_DIR = "saved_models"
SAVE_DIR_SCORES = "PneumoniaMNIST/results_PneumoniaMNIST_scores"

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
PNEUMONIAMNIST = "pneumonia"
AGGREGATED_HEATMAP = "aggr"
CHANNEL_HEATMAP = "chan"
CHANNEL_LEVEL_HEATMAP = "chan_level"
TITLE_GH_POS = "Positive clauses"
TITLE_GH_NEG = "Negative clauses"
RES_HEATMAP = "res"