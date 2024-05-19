abc_path = '/Users/aleksandr/Desktop/abc/abc'
lib_path = '/Users/aleksandr/Desktop/abc/nangate45.lib'
benchmarks_dir = '/Users/aleksandr/Desktop/benches_learn/'
predict_path = '/Users/aleksandr/Desktop/benches_predict/'

state_size = 11  # Размер вектора состояния
action_size = 3  # Размер пространства действий (refactor, rewrite, balance)
learning_rate = 0.001
gamma = 0.9
epsilon = 0.7
epsilon_decay = 0.95
epsilon_min = 0.3
episodes = 100
batch_size = 10s
max_steps_per_episode = 10