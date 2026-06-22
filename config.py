import torch
import os


class Config:
    def __init__(self):
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 模型和训练参数
        self.OBS_LEN = 8
        self.PRED_LEN = 12
        self.BATCH_SIZE = 64
        self.NUM_EPOCHS = 300
        self.LEARNING_RATE = 1e-4
        self.EARLY_STOPPING_PATIENCE = 20
        self.EARLY_STOPPING_DELTA = 1e-6
        self.MODEL_SAVE_PATH = 'model.pth' 
        self.CHECKPOINT_DIR = 'checkpoints' 

        # --- 新增：学习率调度器参数 ---
        self.LR_SCHEDULER_MODE = 'min' 
        self.LR_SCHEDULER_FACTOR = 0.5 
        self.LR_SCHEDULER_PATIENCE = 8 
        self.LR_SCHEDULER_THRESHOLD = 1e-6 
        self.LR_SCHEDULER_MIN_LR = 1e-6 

        self.IMAGE_H = 64
        self.IMAGE_W = 64
        self.IMAGE_C = 13

        self.ENV_FEATURE_DIMS = {
            'wind': 1, 'intensity_class': 6, 'move_velocity': 1, 'month': 12,
            'location_long': 36, 'location_lat': 12, 'history_direction12': 8,
            'history_direction24': 8, 'history_inte_change24': 4
        }
        
        self.EXTRA_DATA_DIM = 1

        self.OUTPUT_DIM = 2
        self.HIDDEN_DIM = 256
        self.NUM_GS = 6
        self.NUM_SAMPLES = 6

        # 数据路径
        self.data_root = '/public/TCND_tcf/'

        # DummyArgs 中的参数，直接作为 Config 的属性
        self.obs_len = self.OBS_LEN # 保持一致性
        self.pred_len = self.PRED_LEN # 保持一致性
        self.skip = 1
        self.delim = '\t'
        self.other_modal = 'gph'
        self.areas = ['WP']
        self.loader_num_workers = 64 # DataLoader 的 worker 数量

# 创建一个 Config 实例
config = Config()

# 确保检查点目录存在
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
