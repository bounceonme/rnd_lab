# Publication Checklist

Use this checklist before pushing the fork to a public GitHub repository.

## Repository Identity

- Choose a repository name that does not imply this is the official upstream
  `fan-ziqi/robot_lab` project.
- Keep the upstream attribution in `README.md`, `NOTICE`, and `LICENSE`.
- Replace placeholder clone URLs in `README.md` with the actual `rnd_lab` public fork URL.
- If `source/robot_lab/config/extension.toml` is updated for publication, use
  the fork URL and fork maintainer name while preserving upstream attribution.

## Public Scope

- Present the maintained target as the non-4bar RND STEP humanoid environment.
- Do not claim maintainership of upstream `robot_lab`.
- Do not list upstream non-RND robot environments as maintained targets unless
  they are actively supported in this fork.
- Keep the task table focused on:
  - `RNDLab-Isaac-Velocity-Flat-RND-Step-v0`
  - `RNDLab-Isaac-Velocity-Rough-RND-Step-v0`

## Files To Exclude

Do not publish local/generated files such as:

- `.codex/`
- `.vscode/browse.vc.db*`
- `*.bak`
- `*.log`
- `*.zip`
- `train_log/`
- `runs/`, `logs/`, `outputs/`, `videos/`, `wandb/`

If any of these are already staged, remove them from the Git index before
committing.

## Minimal Public Evidence

For a credible OSS application, include:

- a focused README;
- clear upstream attribution;
- this changelog;
- environment design documentation;
- installation and train/play commands;
- a short issue or roadmap list for future STEP work;
- at least one screenshot or video link after runtime validation.

## Suggested Codex For OSS Positioning

Use wording close to:

> I maintain `rnd_lab`, a public fork of `fan-ziqi/robot_lab` focused on RND STEP humanoid
> sim-to-real reinforcement learning. The fork maintains STEP robot asset
> integration, Isaac Lab velocity tasks, reward/randomization profiles, and
> train/play workflows for policy development and real-robot transfer.

Avoid wording that implies you are an upstream `robot_lab` core maintainer.
