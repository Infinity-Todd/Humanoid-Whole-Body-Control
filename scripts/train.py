import argparse
from functools import partial
from pathlib import Path
import pickle, shutil
VALID_ENVS = ["humanoid_walk", "humanoid_reach", "humanoid_pick", "humanoid_walkpick"]
def import_env(env_name:str):
    if env_name == "humanoid_walk":
        from envs.humanoid_walk import HumanoidWalkEnv as Env
    elif env_name == "humanoid_reach":
        from envs.humanoid_reach import HumanoidReachEnv as Env
    elif env_name == "humanoid_pick":
        from envs.humanoid_pick import HumanoidPickEnv as Env
    elif env_name == "humanoid_walkpick":
        from envs.humanoid_walkpick import HumanoidWalkPickEnv as Env
    else:
        raise ValueError(
            f"Unknown env name '{env_name}'. "
            "Choose from: humanoid_walk, humanoid_reach, humanoid_pick, humanoid_walkpick"
        )
    return Env
def parse_args():
    ap = argparse.ArgumentParser("Train humanoid whole-body control")
    # 你项目里支持的环境名（此步只解析字符串，不会去真正创建环境）
    ap.add_argument("--env", required=True, choices=VALID_ENVS,
                    help="which environment to use")
    # 可选：把一个 YAML 配置路径传进环境构造器（现在先传，不在这里用）
    ap.add_argument("--yaml", default=None, help="path to task/env yaml (optional)")
    # 以后会接“对称包装”（镜像增强）；现在只占位
    ap.add_argument("--no-mirror", action="store_true", help="disable symmetric wrapper (placeholder)")
    # 日志目录（后面训练时会用来存超参/ckpt）
    ap.add_argument("--logdir", default="logs/run", help="where to save logs/checkpoints later")
    # 其它参数先占坑，后面我们逐步接上
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--total-steps", type=int, default=2_000_000)
    ap.add_argument("--save-path", default="models/model.zip")
    return ap.parse_args()
def run_experiment(args):
    Env = import_env(args.env)
    env_fn = partial(Env, path_to_yaml=args.yaml)
    return env_fn, Env
def dump_experiment_metadata(args):
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    with open(logdir / "experiment.pkl", "wb") as f:   # 2) 保存参数快照
        pickle.dump(args, f)
    if args.yaml:                                      # 3) 复制配置文件
        shutil.copyfile(args.yaml, logdir / "config.yaml")
if __name__ == "__main__":
    args = parse_args()
    env_fn, Env = run_experiment(args)
    dump_experiment_metadata(args)
    print(f"[OK] parsed env: {Env.__name__} | yaml={args.yaml} | logdir={args.logdir}")