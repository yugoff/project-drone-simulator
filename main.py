from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src.drone_env import DroneEnv

def main():
    env = DroneEnv()
    check_env(env)
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=1, 
        learning_rate=1e-4,  # Уменьшенная скорость обучения
        n_steps=100,  # Увеличьте количество шагов для более стабильного обучения
        batch_size=64,  # Увеличьте размер батча для улучшенной статистики
        ent_coef=0.01,  # Энтропийный коэффициент для более разнообразных действий
        gamma=0.99,  # Меньше скидка на будущие вознаграждения
        gae_lambda=0.95  # Гамма для улучшенной оценки будущих вознаграждений
    )
    model.learn(total_timesteps=100000, log_interval=10)
    model.save('drone_model')

    obs, _ = env.reset()
    for _ in range(100000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

if __name__ == '__main__':
    main()