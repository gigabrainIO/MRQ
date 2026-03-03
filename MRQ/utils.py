# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
import pprint

import numpy as np


def enforce_dataclass_type(dataclass: dataclasses.dataclass):
    for field in dataclasses.fields(dataclass):
        setattr(dataclass, field.name, field.type(getattr(dataclass, field.name)))


def set_instance_vars(hp: dataclasses.dataclass, c: object):
    for field in dataclasses.fields(hp):
        c.__dict__[field.name] = getattr(hp, field.name)


class Logger:
    def __init__(self, log_file: str, enable_wandb: bool = False, wandb_config: dict = None):
        """
        A logger that prints to file + console, and (optionally) logs to Weights & Biases.

        Args:
            log_file (str): Path to the log file.
            enable_wandb (bool): Whether to log to wandb.
            wandb_config (dict): Config for wandb.init() (project, entity, etc).
        """
        self.log_file = log_file
        self.enable_wandb = enable_wandb
        self.wandb_initialized = False

        # Initialize wandb if requested
        if self.enable_wandb:
            self.init_wandb(wandb_config or {})

    def init_wandb(self, config: dict):
        """
        Initialize wandb logging.

        Example:
            logger.init_wandb({
                "project": "mrq_experiments",
                "entity": "my_team",
                "name": "run1"
            })
        """
	# Try importing wandb, otherwise define a dummy
	try:
	    import wandb
	    self.wandb_found = True
	except ImportError:
	    wandb = None
	    self.wandb_found = False
        if not self.wandb_found:
            self.log_print("WARNING: wandb not installed — wandb logging disabled")
            return

        try:
            self.wandb_run = wandb.init(**config)
            self.wandb_initialized = True
            self.log_print(f"WandB run initialized: {self.wandb_run.name}")
        except Exception as e:
            self.log_print(f"Failed to init wandb: {e}")
            self.enable_wandb = False

    def log_print(self, x: str | object):
        with open(self.log_file, 'a') as f:
            if isinstance(x, str):
                print(x)
                f.write(x+'\n')
            else:
                pprint.pprint(x)
                pprint.pprint(x, f)


    def title(self, text: str):
        self.log_print('-'*40)
        self.log_print(text)
        self.log_print('-'*40)

    def log_metrics(self, metrics: dict, step: int = None):
        """
        Log a dictionary of scalars.

        Args:
            metrics (dict): Keys → metric names, values → scalars.
            step (int | None): Optional global step for wandb/log prints.
        """
        # Log locally
        msg = f"[STEP {step}] " + " | ".join([f"{k}: {v}" for k, v in metrics.items()])
        self.log_print(msg)

        # Log to wandb if enabled
        if self.enable_wandb and self.wandb_initialized:
            try:
                if step is not None:
                    wandb.log(metrics, step=step)
                else:
                    wandb.log(metrics)
            except Exception as e:
                self.log_print(f"WARNING: failed to wandb.log metrics: {e}")

    def log_scalar(self, name: str, value: float, step: int = None):
        """
        Log a single scalar value.
        """
        self.log_metrics({name: value}, step)

    def log_gradients(self, model, prefix="grad"):
        """Log gradient norms for a PyTorch model."""
        if not self.enable_wandb or not self.wandb_initialized:
            return
        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[f"{prefix}/{name}"] = param.grad.norm().item()
        wandb.log(grads)

    def log_model_weights(self, model, prefix="weight"):
        """Log model weight norms."""
        if not self.enable_wandb or not self.wandb_initialized:
            return
        weights = {}
        for name, param in model.named_parameters():
            weights[f"{prefix}/{name}"] = param.data.norm().item()
        wandb.log(weights)

    def log_artifact(self, artifact_path: str, artifact_name: str, artifact_type="model"):
        """Log a file/artifact to wandb."""
        if self.enable_wandb and self.wandb_initialized:
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)

# Takes the formatted results and returns a dictionary of env -> (timesteps, seed).
def results_to_numpy(file: str='../results/gym_results.txt'):
    results = {}

    for line in open(file):
        if '----' in line:
            continue
        if 'Timestep' in line:
            continue
        if 'Env:' in line:
            env = line.split(' ')[1][:-1]
            results[env] = []
        else:
            timestep = []
            for seed in line.split('\t')[1:]:
                if seed != '':
                    seed = seed.replace('\n', '')
                    timestep.append(float(seed))
            results[env].append(timestep)

    for k in results:
        results[k] = np.array(results[k])
        print(k, results[k].shape)

    return results


gym = [
    'Gym-HalfCheetah-v4',
    'Gym-Hopper-v4',
    'Gym-Walker2d-v4',
    'Gym-Ant-v4',
    'Gym-Humanoid-v4',
]


dmc = [
    'Dmc-acrobot-swingup',
    'Dmc-ball_in_cup-catch',
    'Dmc-cartpole-balance',
    'Dmc-cartpole-balance_sparse',
    'Dmc-cartpole-swingup',
    'Dmc-cartpole-swingup_sparse',
    'Dmc-cheetah-run',
    'Dmc-dog-stand',
    'Dmc-dog-walk',
    'Dmc-dog-trot',
    'Dmc-dog-run',
    'Dmc-finger-spin',
    'Dmc-finger-turn_easy',
    'Dmc-finger-turn_hard',
    'Dmc-fish-swim',
    'Dmc-hopper-stand',
    'Dmc-hopper-hop',
    'Dmc-humanoid-stand',
    'Dmc-humanoid-walk',
    'Dmc-humanoid-run',
    'Dmc-pendulum-swingup',
    'Dmc-quadruped-walk',
    'Dmc-quadruped-run',
    'Dmc-reacher-easy',
    'Dmc-reacher-hard',
    'Dmc-walker-stand',
    'Dmc-walker-walk',
    'Dmc-walker-run'
]


dmc_visual = [
    'Dmc-visual-acrobot-swingup',
    'Dmc-visual-ball_in_cup-catch',
    'Dmc-visual-cartpole-balance',
    'Dmc-visual-cartpole-balance_sparse',
    'Dmc-visual-cartpole-swingup',
    'Dmc-visual-cartpole-swingup_sparse',
    'Dmc-visual-cheetah-run',
    'Dmc-visual-dog-stand',
    'Dmc-visual-dog-walk',
    'Dmc-visual-dog-trot',
    'Dmc-visual-dog-run',
    'Dmc-visual-finger-spin',
    'Dmc-visual-finger-turn_easy',
    'Dmc-visual-finger-turn_hard',
    'Dmc-visual-fish-swim',
    'Dmc-visual-hopper-stand',
    'Dmc-visual-hopper-hop',
    'Dmc-visual-humanoid-stand',
    'Dmc-visual-humanoid-walk',
    'Dmc-visual-humanoid-run',
    'Dmc-visual-pendulum-swingup',
    'Dmc-visual-quadruped-walk',
    'Dmc-visual-quadruped-run',
    'Dmc-visual-reacher-easy',
    'Dmc-visual-reacher-hard',
    'Dmc-visual-walker-stand',
    'Dmc-visual-walker-walk',
    'Dmc-visual-walker-run'
]


atari = [
    'Atari-Alien-v5',
    'Atari-Amidar-v5',
    'Atari-Assault-v5',
    'Atari-Asterix-v5',
    'Atari-Asteroids-v5',
    'Atari-Atlantis-v5',
    'Atari-BankHeist-v5',
    'Atari-BattleZone-v5',
    'Atari-BeamRider-v5',
    'Atari-Berzerk-v5',
    'Atari-Bowling-v5',
    'Atari-Boxing-v5',
    'Atari-Breakout-v5',
    'Atari-Centipede-v5',
    'Atari-ChopperCommand-v5',
    'Atari-CrazyClimber-v5',
    'Atari-DemonAttack-v5',
    'Atari-DoubleDunk-v5',
    'Atari-Enduro-v5',
    'Atari-FishingDerby-v5',
    'Atari-Freeway-v5',
    'Atari-Frostbite-v5',
    'Atari-Gopher-v5',
    'Atari-Gravitar-v5',
    'Atari-Hero-v5',
    'Atari-IceHockey-v5',
    'Atari-Jamesbond-v5',
    'Atari-Kangaroo-v5',
    'Atari-Krull-v5',
    'Atari-KungFuMaster-v5',
    'Atari-MontezumaRevenge-v5',
    'Atari-MsPacman-v5',
    'Atari-NameThisGame-v5',
    'Atari-Phoenix-v5',
    'Atari-Pitfall-v5',
    'Atari-Pong-v5',
    'Atari-PrivateEye-v5',
    'Atari-Qbert-v5',
    'Atari-Riverraid-v5',
    'Atari-RoadRunner-v5',
    'Atari-Robotank-v5',
    'Atari-Seaquest-v5',
    'Atari-Skiing-v5',
    'Atari-Solaris-v5',
    'Atari-SpaceInvaders-v5',
    'Atari-StarGunner-v5',
    'Atari-Tennis-v5',
    'Atari-TimePilot-v5',
    'Atari-Tutankham-v5',
    'Atari-UpNDown-v5',
    'Atari-Venture-v5',
    'Atari-VideoPinball-v5',
    'Atari-WizardOfWor-v5',
    'Atari-YarsRevenge-v5',
    'Atari-Zaxxon-v5',
]
