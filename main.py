import config
import environment1
import DQN_model
import os

bench_files = [os.path.join(config.benchmarks_dir, bench) for bench in os.listdir(config.benchmarks_dir) if bench.endswith("_orig.bench")]

agent = DQN_model.DQNAgent(DQN_model.state_size, DQN_model.action_size, DQN_model.learning_rate, DQN_model.gamma, DQN_model.epsilon, DQN_model.epsilon_decay, DQN_model.epsilon_min)

for bench_file in bench_files:
    print(f"Обучение на файле {bench_file}")
    env = environment1.ABCOptimizationEnv(config.abc_path, bench_file, config.lib_path)
    env.parse_stats()
    env.create_new_file()
    DQN_model.train_dqn(agent, env, DQN_model.episodes, DQN_model.batch_size, DQN_model.max_steps_per_episode)

stats = DQN_model.predict_and_optimize(agent, config.predict_path)

with open('model_stats.txt', 'w', encoding='utf-8') as f:
    for key, value in stats.items():
        f.write(f'{key}: {value}\n')