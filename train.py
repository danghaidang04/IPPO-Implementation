# file: train.py

import time
from algs.ippo.ippo_trainer import IPPOTrainer
from env import SimpleTagEnv # Import env từ câu trả lời trước
from config import MODELS_PATH
def main():

    # Khởi tạo môi trường
    print("Khởi tạo môi trường...")
    env = SimpleTagEnv(grid_size=10, max_steps=100)

    # --- Hyperparameters ---
    NUM_EPISODES = 100         # Tổng số episodes để huấn luyện
    ROLLOUT_STEPS = 128         # Số bước thu thập dữ liệu trước mỗi lần update
    
    ACTOR_LR = 3e-4             # Learning rate cho Actor
    CRITIC_LR = 1e-3            # Learning rate cho Critic
    GAMMA = 0.99                # Discount factor
    GAE_LAMBDA = 0.95           # Lambda cho GAE
    PPO_CLIP = 0.2              # Ngưỡng clip của PPO
    K_EPOCHS = 4                # Số epochs update trên mỗi batch
    ENTROPY_COEFF = 0.01        # Hệ số entropy (khuyến khích khám phá)
    # -------------------------

    print("Khởi tạo IPPOTrainer...")
    trainer = IPPOTrainer(
        env=env,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        ppo_clip=PPO_CLIP,
        k_epochs=K_EPOCHS,
        entropy_coeff=ENTROPY_COEFF,
        use_gpu=True
    )

    print("Bắt đầu huấn luyện...")
    start_time = time.time()
    
    total_steps = 0
    
    # Sử dụng `num_episodes` để dễ theo dõi
    for episode in range(1, NUM_EPISODES + 1):
        
        # Hàm learn sẽ tự động chạy rollout `ROLLOUT_STEPS` bước
        # và sau đó thực hiện cập nhật
        trainer.learn(num_rollout_steps=ROLLOUT_STEPS)
        total_steps += ROLLOUT_STEPS

        if episode % 100 == 0:
            print("---------------------------------")
            print(f"Episode: {episode} / {NUM_EPISODES}")
            print(f"Tổng số bước: {total_steps}")
            print(f"Thời gian: {(time.time() - start_time) / 60:.2f} phút")
            
            # Bạn có thể thêm code để đánh giá (evaluation) ở đây
            # ...
            
    print("Huấn luyện hoàn tất!")
    env.close()

    # Lưu mô hình sau khi huấn luyện
    trainer.save_models(MODELS_PATH)

if __name__ == "__main__":
    main()