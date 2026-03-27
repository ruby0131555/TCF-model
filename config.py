import torch
import os

class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Sequence and training hyperparameters
        self.OBS_LEN = 8
        self.PRED_LEN = 12
        self.BATCH_SIZE = 128
        self.NUM_EPOCHS = 500
        self.LEARNING_RATE = 1e-5
        
        # Early stopping configuration
        self.EARLY_STOPPING_PATIENCE = 30
        self.EARLY_STOPPING_DELTA = 1e-6
        
        # Model saving paths
        self.MODEL_SAVE_PATH = 'model_tcf.pth' 
        self.CHECKPOINT_DIR = 'checkpoints' 

        # Learning rate scheduler settings
        self.LR_SCHEDULER_MODE = 'min' 
        self.LR_SCHEDULER_FACTOR = 0.5 
        self.LR_SCHEDULER_PATIENCE = 8 
        self.LR_SCHEDULER_THRESHOLD = 1e-6 
        self.LR_SCHEDULER_MIN_LR = 1e-6 

        # 3D Image input dimensions
        self.IMAGE_H = 64
        self.IMAGE_W = 64
        self.IMAGE_C = 13

        # Dimensions for environmental feature dictionary
        self.ENV_FEATURE_DIMS = {
            'wind': 1, 'intensity_class': 6, 'move_velocity': 1, 'month': 12,
            'location_long': 36, 'location_lat': 12, 'history_direction12': 8,
            'history_direction24': 8, 'history_inte_change24': 4
        }
        
        # Model architecture and output settings
        self.OUTPUT_DIM = 2
        self.HIDDEN_DIM = 256
        self.NUM_SAMPLES = 6

        self.data_root = '/public/liaoyr/TCND_tcf/'
        self.obs_len = self.OBS_LEN 
        self.pred_len = self.PRED_LEN 
        self.skip = 1
        self.delim = '\t'
        self.other_modal = 'gph'
        self.areas = ['WP']
        self.loader_num_workers = 32 

config = Config()

os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
