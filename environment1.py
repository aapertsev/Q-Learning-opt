import gym
from gym import spaces
import subprocess
import numpy as np
import time

class ABCOptimizationEnv(gym.Env):
    def __init__(self, abc_path, benches_path, lib_path):
        super(ABCOptimizationEnv, self).__init__()
        self.abc_path = abc_path
        self.benches_path = benches_path
        self.lib_path = lib_path
        self.action_space = spaces.Discrete(3)  # refactor, rewrite, balance
        self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.metrics = np.zeros(2) #1 - задержка, 2 - площадь
        self.history = []
        self.ACTION_MAPPING = {
            0: 'refactor',
            1: 'rewrite',
            2: 'balance'
        }

    def create_new_file(self):
        optimized_bench_path = self.benches_path.replace(".bench", "_optimized.bench")

        # Копирование содержимого исходного файла в новый файл с припиской '_optimized'
        with open(self.benches_path, 'r') as original, open(optimized_bench_path, 'w') as optimized:
            optimized.write(original.read())

        # Обновление пути к bench файлу, чтобы в дальнейшем использовать оптимизированный файл
        self.benches_path = optimized_bench_path

    def parse_stats(self):
        start_time = time.time()
        # Формирование команды для abc
        abc_command = f'{self.abc_path} -c "read_bench {self.benches_path}; read {self.lib_path}; map; print_stats "'

        # Выполнение команды и получение вывода
        process = subprocess.Popen(abc_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, _ = process.communicate()
        s = stdout.decode('utf-8')  # Декодирование вывода в строку
        #print(s)

        numbers = []  # Массив для хранения извлеченных чисел
        i = 0  # Индекс текущего символа в строке

        while i < len(s):
            if s[i] == '=' or s[i] =='/':  # Поиск знака '='
                i += 1  # Пропустить знак '='
                number_str = ''  # Строка для сбора символов текущего числа

                # Пропускаем пробелы после знака '='
                while i < len(s) and s[i] == ' ':
                    i += 1

                # Собрать число
                while i < len(s) and (s[i].isdigit() or s[i] == '.'):
                    number_str += s[i]
                    i += 1

                if number_str:  # Если строка с числом не пуста
                    numbers.append(float(number_str))  # Преобразовать в число и добавить в массив

            else:
                i += 1  # Переход к следующему символу, если текущий не '='
        numbers.pop(0)

        # Извлечение значений из строки вывода
        inputs = int(numbers[0])
        outputs = int(numbers[0])
        #outputs = int(numbers[1])
        lat = int(numbers[2])
        nd = int(numbers[3])
        edge = int(numbers[4])
        lev = int(numbers[7])
        area = float(numbers[5])
        delay = float(numbers[6])

        # Обновление вектора состояния и метрик
        state = np.array([inputs, outputs, lat, nd, edge, lev, area, delay, self.state[8], self.state[9], self.state[10]], dtype=np.float32)
        #print(state)

        state[0:8] = state[0:8]/np.linalg.norm(state[0:8])
        #if(np.linalg.norm(state[8:])!=0):
            #state[8:] = state[8:]/np.linalg.norm(state[8:])
        self.state = state
        print(self.state)
        self.metrics = np.array([delay, area], dtype=np.float32)
        print(self.metrics)
        print(f"Статистика разобрана за {time.time() - start_time:.2f} секунд")
        return self.metrics, self.state




    def step(self, action):
        start_time = time.time()
        action_str = self.ACTION_MAPPING[action]
        abc_command = f'{self.abc_path} -c "read_bench {self.benches_path}; read {self.lib_path};strash; {action_str};  write_bench {self.benches_path}"'

        # Выполнение команды с перенаправлением stderr в stdout, чтобы захватить возможные ошибки
        process = subprocess.run(abc_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Проверка на наличие ошибок и их вывод
        if process.returncode != 0:  # Если процесс завершился с кодом ошибки
            print(f"Error executing ABC command: {process.stdout}")
        #print('1:', self.metrics)

        last_metrics = self.metrics
        last_state = self.state

        self.parse_stats()

        #print('2:',last_metrics)
        #print('3:',self.metrics)

        adp = self.metrics[0]*self.metrics[1]
        adp_last = last_metrics[0]*last_metrics[1]

        if adp<=adp_last:
            reward = (1-adp/adp_last)*10
        if adp/adp_last>1:
            reward = -(adp/adp_last-1)*10

        print(f"Действие {action_str} выполнено за {time.time() - start_time:.2f} секунд")
        print(f"Изменение метрик: задержка с {last_metrics[0]} до {self.metrics[0]}, площадь с {last_metrics[1]} до {self.metrics[1]}")
        print(f"Награда: {reward}")

        return last_state, reward

    def reset(self):
        self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.metrics = np.zeros(2) #1 - задержка, 2 - площадь
        self.history = []

    def render(self):
        print(f"Текущее состояние: {self.state}")

    def close(self):
        pass

