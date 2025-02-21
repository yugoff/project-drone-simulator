from stable_baselines3 import PPO
import gymnasium as gym
import pybullet_data
import time
import numpy as np
import pybullet as p
from src.drone_env import DroneEnv  # Убедитесь, что правильный путь к вашему файлу

def test_model():
    # Загружаем обученную модель
    model = PPO.load('drone_model')  # Здесь указать путь к вашей сохраненной модели

    # Создаем среду
    env = DroneEnv()
    env.reset()

    # Протестируем модель
    obs, _ = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)  # Модель предсказывает действия
        obs, reward, terminated, truncated, info = env.step(action)  # Применяем действия в среде
        env.render()  # Визуализируем результат
        print(f'Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}')  # Выводим награду

        # Завершаем тестирование, если условия выполнены
        if terminated or truncated:
            print('Test finished!')
            break

    env.close()  # Закрываем среду

if __name__ == '__main__':
    test_model()