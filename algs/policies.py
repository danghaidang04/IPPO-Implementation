# file: algs/policies.py
import torch
import os

# --- [ 1. SỬA DÒNG NÀY ] ---
# Import Class 'Actor' từ file 'actor.py' của bạn
from networks.actor import Actor # <-- Đảm bảo đường dẫn này ĐÚNG

# --- [ 2. SỬA DÒNG NÀY ] ---
# Trỏ đến ĐÚNG 'path_prefix' mà bạn đã dùng để lưu model
MODEL_PREFIX_PATH = "./models/ippo/ippo"  # <-- SỬA ĐÚNG TÊN FILE

# --- Biến Toàn cục (Global Cache) ---
_LOADED_ACTORS_CACHE = {}
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_models_to_cache(env):
    """
    Hàm helper, tải file .pth vào cache.
    """
    global _LOADED_ACTORS_CACHE
    print(f"[Policies] Đang tải model IPPO từ prefix: {MODEL_PREFIX_PATH}...")
    
    try:
        agent_sample = env.possible_agents[0]
        state_dim = env.observation_space(agent_sample).shape[0]
        action_dim = env.action_space(agent_sample).n

        loaded_models = {}
        for agent_id in env.possible_agents:
            # 1. Tạo Actor rỗng (từ bản thiết kế)
            actor_model = Actor(state_dim, action_dim).to(_DEVICE)
            
            # 2. Tìm đường dẫn file .pth
            actor_path = f"{MODEL_PREFIX_PATH}_actor_{agent_id}.pth"
            if not os.path.exists(actor_path):
                raise FileNotFoundError(f"Không tìm thấy file model: {actor_path}")

            # 3. Load tham số (file .pth) vào Actor rỗng
            actor_model.load_state_dict(torch.load(actor_path, map_location=_DEVICE))
            actor_model.eval()
            
            # 4. Lưu vào cache
            loaded_models[agent_id] = actor_model
            
        _LOADED_ACTORS_CACHE = loaded_models
        print(f"[Policies] Tải thành công {len(_LOADED_ACTORS_CACHE)} actors vào cache.")
        
    except Exception as e:
        print(f"[Policies] LỖI KHI LOAD MODEL: {e}")
        raise e

# --- HÀM CHÍNH (ĐÃ SỬA) ---

def get_actions_policy(algorithm_name, env, observations):
    actions = {}
    
    if algorithm_name == 'Random':
        for agent_id in env.agents:
            actions[agent_id] = env.action_space(agent_id).sample()
    
    elif algorithm_name == 'IPPO':
        # 1. Kiểm tra cache
        if not _LOADED_ACTORS_CACHE:
            _load_models_to_cache(env) # Tải model (chỉ 1 lần)
        
        # 2. Sử dụng model từ cache
        for agent_id in env.agents:
            if agent_id not in observations:
                continue
            
            # Lấy actor (đã load) từ cache
            actor = _LOADED_ACTORS_CACHE[agent_id]
            
            state_tensor = torch.tensor(observations[agent_id], dtype=torch.float32).to(_DEVICE)
            
            with torch.no_grad():
                action, _ = actor.get_action(state_tensor)
                actions[agent_id] = action

    else:
        raise ValueError(f"Thuật toán '{algorithm_name}' không được hỗ trợ.")
    
    return actions