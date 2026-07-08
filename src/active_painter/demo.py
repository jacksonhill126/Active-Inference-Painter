from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .agent import ActiveInferencePainter
from .config import PainterConfig
from .env import PaintCanvasEnv, StrokeAction
from .policies import PolicySampler


def pretrain(agent: ActiveInferencePainter, env: PaintCanvasEnv, steps: int) -> None:
    sampler = PolicySampler(agent.cfg, seed=991)
    for i in range(steps):
        action = sampler._stroke()  # Synthetic motor babbling in the generative process.
        if env.done or env.latent_state()[0] > 0.96 or i % 35 == 34:
            env.reset()
        agent.collect_random_transition(env, action)
        if i > agent.cfg.batch_size and i % 4 == 0:
            agent.train_dynamics(gradient_steps=2)
    for _ in range(250):
        agent.train_dynamics(gradient_steps=1)


def save_canvas(env: PaintCanvasEnv, output: Path, episode: int) -> None:
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.imshow(env.observed_tone(), cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(f"Episode {episode} · coverage={env.latent_state()[0]:.3f}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / f"episode_{episode:03d}.png", dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    cfg = PainterConfig(
        candidate_policies=args.candidates,
        planning_horizon=args.horizon,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    env = PaintCanvasEnv(cfg, seed=args.seed)
    agent = ActiveInferencePainter(cfg, seed=args.seed, device=args.device)
    pretrain(agent, env, args.pretrain_steps)

    trace: list[dict[str, object]] = []
    for episode in range(args.episodes):
        observation = env.reset()
        agent.reset_belief(observation)
        for decision in range(args.max_decisions):
            policy, chosen, ranked = agent.infer_policy()
            action = policy.actions[0]
            before = env.latent_state().copy()
            observation, done, info = env.step(action)
            after = env.latent_state().copy()
            if not action.stop:
                agent.replay.add(before, action.vector(), after)
                agent.train_dynamics(gradient_steps=4)
                agent.update_belief(action, observation)

            trace.append(
                {
                    "episode": episode,
                    "decision": decision,
                    "action": agent.policy_dict(policy)[0],
                    "chosen_policy": agent.policy_dict(policy),
                    "efe": asdict(chosen),
                    "coverage": float(after[0]),
                    "stop_posterior": float(
                        next(prob for p, _, prob in ranked if len(p.actions) == 1 and p.actions[0].stop)
                    ),
                    "info": info,
                }
            )
            print(
                f"ep={episode:02d} step={decision:03d} coverage={after[0]:.3f} "
                f"action={'STOP' if action.stop else 'stroke'} "
                f"G={chosen.total:.3f} terminal={chosen.terminal_coverage_mean:.3f}±{chosen.terminal_coverage_std:.3f}"
            )
            if done:
                break
        save_canvas(env, output, episode)

    (output / "trace.json").write_text(json.dumps(trace, indent=2))
    print(f"Wrote outputs to {output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--pretrain-steps", type=int, default=1200)
    parser.add_argument("--max-decisions", type=int, default=80)
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="runs")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
