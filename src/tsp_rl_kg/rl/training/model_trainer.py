class ModelTrainer:
    def __init__(self, env, eval_env, logger, device):
        self.env = env
        self.eval_env = eval_env
        self.logger = logger
        self.device = device
        self.rl_model = None
        self.metrics = TrainingMetrics(env.action_space.n)

    def create_model(self, model_config):
        self.logger.info("Creating PPO model", logger_name='training')
        self.rl_model = PPO("MultiInputPolicy", 
                    self.env, 
                    policy_kwargs={
                        'features_extractor_class': AgentModel,
                        'features_extractor_kwargs': {'features_dim': 64}
                    },
                    **model_config,
                    device=self.device,
                    verbose=1
        )
        self.logger.info("PPO model created successfully", logger_name='training')

    def train(self, total_timesteps, eval_callback, timeout=3600):
        self.logger.info("Starting model training", logger_name='training')
        
        curriculum_callback = CurriculumCallback(self.eval_env, self.logger, self.metrics)
        
        try:
            self.rl_model.learn(
                total_timesteps=total_timesteps,
                callback=[eval_callback, curriculum_callback],
                reset_num_timesteps=False,
                tb_log_name="PPO",
                progress_bar=True
            )
        except Exception as e:
            self.logger.error(f"An error occurred during training: {str(e)}")
            self.logger.error(traceback.format_exc())
        
        if curriculum_callback.should_stop:
            self.logger.info("Training ended early due to early stop condition", logger_name='training')
        else:
            self.logger.info("Model training completed", logger_name='training')
        
        # Save metrics after training
        self.metrics.save_to_csv("training_metrics.csv")

    def log_training_stats(self):
        mean_reward = self.calculate_mean_reward()
        mean_episode_length = self.calculate_mean_episode_length()
        self.logger.info(f"Recent mean reward: {mean_reward:.2f}", logger_name='training')
        self.logger.info(f"Recent mean episode length: {mean_episode_length:.2f}", logger_name='training')
        self.logger.info(f"Episode info buffer size: {len(self.rl_model.ep_info_buffer)}", logger_name='training')
        if len(self.rl_model.ep_info_buffer) > 0:
            self.logger.info(f"Sample episode info: {self.rl_model.ep_info_buffer[-1]}", logger_name='training')
        self.logger.info(f"Recent policy loss: {self.rl_model.logger.name_to_value['train/policy_loss']:.5f}", logger_name='training')
        self.logger.info(f"Recent value loss: {self.rl_model.logger.name_to_value['train/value_loss']:.5f}", logger_name='training')

    def calculate_mean_reward(self):
        if len(self.rl_model.ep_info_buffer) > 0:
            return np.mean([ep_info["r"] for ep_info in self.rl_model.ep_info_buffer])
        return 0.0

    def calculate_mean_episode_length(self):
        if len(self.rl_model.ep_info_buffer) > 0:
            return np.mean([ep_info["l"] for ep_info in self.rl_model.ep_info_buffer])
        return 0.0

    def save_model(self, path):
        self.logger.info(f"Saving model to {path}", logger_name='training')
        self.rl_model.save(path)
        self.logger.info("Model saved successfully", logger_name='training')

    def evaluate_model(self, eval_env, n_eval_episodes=10):
        self.logger.info("Starting final model evaluation", logger_name='eval')
        
        episode_rewards = []
        for _ in range(n_eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.rl_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            episode_rewards.append(episode_reward)
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        self.logger.info(f"Final evaluation: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}", logger_name='eval')
        return mean_reward, std_reward