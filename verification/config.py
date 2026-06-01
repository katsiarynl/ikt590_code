
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
DATA_DIR = "processed_data_2"