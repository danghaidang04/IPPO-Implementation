# config.py

# --- Cấu hình môi trường ---
ENV_NAME = "simple_spread_v3"
NUM_AGENTS = 3 # Môi trường simple_spread_v3 mặc định có 3 agent

# --- Cấu hình huấn luyện ---
STOP_TIMESTEPS = 1_000_000 # Tổng số bước huấn luyện
NUM_WORKERS = 4 # Số CPU core dùng để thu thập dữ liệu song song
ROLLOUT_FRAGMENT_LENGTH = 128 # Số bước mỗi worker thu thập trước khi gửi về learner
LEARNING_RATE = 5e-5
CLIP_PARAM = 0.2
GAMMA = 0.99
TRAIN_BATCH_SIZE = 4096 # Tổng số mẫu dùng trong 1 lần update (lấy từ các worker)

# --- Cấu hình RLlib ---
FRAMEWORK = "torch"