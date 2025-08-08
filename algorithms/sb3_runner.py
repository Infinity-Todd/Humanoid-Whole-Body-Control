from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3 import PPO
import torch
from stable_baselines3.common.utils import get_linear_fn
from pathlib import Path
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
class SB3Runner:
    """负责构建 vecenv → 选设备 → 构建模型 → 回调 → 训练与保存"""
    def __init__(self, args):
        self.args = args
        self._uses_vecnorm= False

    def _build_vec_env(self, env_fn):
        def make(i):
            def _init():
                env= env_fn()
                env.reset(seed=self.args.seed + i)
                try:
                    env = RecordEpisodeStatistics(env)
                except Exception:
                    pass
                return env
            return _init
        vec=SubprocVecEnv([make(i) for i in range(self.args.num_envs)])
        if getattr(self.args, "normalize", False):
            vec = VecNormalize(
                vec,
                norm_obs=True, norm_reward=True,
                clip_obs=10.0, clip_reward=10.0
            )
            self._uses_vecnorm = True
        else:
            self._uses_vecnorm = False
        return vec

    def _pick_device(self):
        pref = (self.args.device or "").lower() if getattr(self.args, "device", None) else None
        try:
            import torch
        except Exception:
            return "cpu"

        if pref in ("cuda", "mps", "cpu"):
            return pref
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _build_model(self, vec_env, device):
        n_envs = max(1, int(getattr(self.args, "num_envs", 1)))
        n_steps = max(128, min(512, 8192 // n_envs))
        policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
        ortho_init=True)
        
        lr_sched = get_linear_fn(3e-4, 3e-5, 1.0)
        
        model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        device=device,
        n_steps=n_steps,
        batch_size=2048,           # 训练小批次；最好能整除 n_envs*n_steps
        n_epochs=10,
        learning_rate=lr_sched,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,            # 防止 PPO 爆走
        use_sde=True,              # 连续控制里常用，增稳
        sde_sample_freq=4,
        tensorboard_log=str(self.args.logdir),
        verbose=1,
        policy_kwargs=policy_kwargs)
        
        return model

    def _build_callbacks(self, eval_env_fn):
        eval_vec = DummyVecEnv([lambda: RecordEpisodeStatistics(eval_env_fn())])
        eval_cb = EvalCallback(
        eval_env=eval_vec,
        best_model_save_path=str(Path(self.args.logdir) / "best"),
        log_path=str(Path(self.args.logdir) / "eval"),
        eval_freq=self.args.eval_every,
        n_eval_episodes=self.args.eval_episodes,
        deterministic=True,
        render=False,
    )
        ckpt_cb = CheckpointCallback(
        save_freq=self.args.checkpoint_freq,
        save_path=str(Path(self.args.logdir) / "checkpoints"),
        name_prefix="ckpt",
        save_replay_buffer=False,
        save_vecnormalize=self._uses_vecnorm,  # 若开启了 VecNormalize，就把统计也保存
    )
        return CallbackList([eval_cb, ckpt_cb])


    def run(self, env_fn):
        device = self._pick_device()
        vec_env = self._build_vec_env(env_fn)
        model = self._build_model(vec_env, device)
        callbacks = self._build_callbacks(env_fn)
        try:
            model.learn(total_timesteps=self.args.total_steps, callback=callbacks)
        except KeyboardInterrupt:
            print("Interrupted by user. Saving partial model...")
        out = Path(self.args.save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
        if self._uses_vecnorm:
            vec_env.save(Path(self.args.logdir)/"vecnormalize.pkl")
        vec_env.close()
        print(f"✅ Done. Saved model to {out}  |  device={device}")
        

