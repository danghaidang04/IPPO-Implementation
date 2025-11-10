from env import SimpleTagEnv
from config import ALGORITHM
from algs.policies import get_actions_policy

if __name__ == "__main__":
    # Chạy kiểm tra API song song của PettingZoo
    print("\nBắt đầu chạy thử (random agents)...")
    env = SimpleTagEnv(grid_size=8, max_steps=50)
    observations, infos = env.reset()
    
    while env.agents:

        # Lấy hành động dựa trên algoritmhm đã chọn
        actions = get_actions_policy(ALGORITHM, env, observations)
        #actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        # Update môi trường với các hành động đã chọn
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # In ra màn hình
        env.render()
        print(f"Step: {env.step_count}")
        print(f"Rewards: {rewards}")
        
        # Dừng một chút để xem
        import time
        time.sleep(0.5)

    print("Hoàn thành chạy thử.")