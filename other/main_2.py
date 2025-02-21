import pybullet
import pybullet_data
import time
import math

# Подключаемся к физическому движку с графическим интерфейсом
pybullet.connect(pybullet.GUI)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
pybullet.setGravity(0, 0, -9.81)

# Загружаем платформу и БЛА
environment = pybullet.loadURDF('environment_old.urdf', basePosition=[0, 0, 0])

# Установим начальные параметры камеры
pybullet.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 0])

# Симуляция
while True:
    pybullet.stepSimulation()

    t = time.time()

    # Настройка платформы

    # Генерируем колебания (по трем осям) для платформы
    # new_x = 0.1 * math.sin(0.5 * t)
    # new_y = 0.1 * math.cos(0.5 * t)
    # new_z = 0.3 + 0.1 * math.sin(2 * t)

    # Покачивание платформы (наклоны)
    # roll = 0.1 * math.sin(0.8 * t)  # Покачивание влево-вправо (наклон по оси X)
    # pitch = 0.1 * math.cos(0.5 * t)  # Покачивание вперёд-назад (наклон по оси Y)
    roll = 0
    pitch = 0

    # Преобразуем углы Эйлера в кватернион для платформы
    orientation = pybullet.getQuaternionFromEuler([roll, pitch, 0])

    # Обновляем позицию и ориентацию платформы
    pybullet.resetBasePositionAndOrientation(environment, [0, 0, 0], orientation)

    # Управление БЛА (подъем и посадка)

    drone_position, _ = pybullet.getLinkState(environment, 1)[:2]

    if drone_position[2] > 0.3:
        new_z = drone_position[2] - 0.005  # Чем меньше значение, тем плавнее спуск
        pybullet.resetBasePositionAndOrientation(environment, [drone_position[0], drone_position[1], new_z], [0, 0, 0, 1])
    else:
        print("Дрон приземлился!")
#     drone_x, drone_y, drone_z = drone_position

#     # Рассчитываем отклонения от целевой высоты
#     height_error = target_position[2] - drone_z

#     # Рассчитываем отклонения по углам платформы
#     roll_error = roll # Угол наклона по оси X платформы
#     pitch_error = pitch # Угол наклона по оси Y платформы

#     # Коэффициенты для коррекции
#     Kp_height = 2 # Пропорциональный коэффициент для высоты
#     Kp_roll = 1 # Пропорциональный коэффициент для отклонения по оси X
#     Kp_pitch = 1 # Пропорциональный коэффициент для отклонения по оси Y

#     # Рассчитываем силу для подъема/опускания дрона
#     thrust = Kp_height * height_error  # Пропорциональная сила по высоте

#     # Коррекция наклонов
#     correction_x = -Kp_roll * roll_error  # Корректировка по оси X
#     correction_y = -Kp_pitch * pitch_error  # Корректировка по оси Y

#     # Применяем внешнюю силу для подъема и коррекции наклонов
    # pybullet.applyExternalForce(drone, -1, [correction_x, correction_y, thrust], drone_position, pybullet.LINK_FRAME)

#     # Ограничение FPS для стабильной симуляции
    time.sleep(1 / 240)