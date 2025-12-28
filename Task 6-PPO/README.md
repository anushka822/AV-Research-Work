# PPO on CarRacing-v2 (Continuous Action Space)

This project implements **Proximal Policy Optimization (PPO)** from scratch using **PyTorch** and **Gymnasium**, applied to the `CarRacing-v2` environment with a continuous action space.

The implementation follows an **Actor-Critic** architecture and includes:
- Generalized Advantage Estimation (GAE)
- Clipped PPO objective
- Reward shaping
- TensorBoard logging

---

## Environment

- **Environment:** `CarRacing-v2`
- **Action Space:** Continuous
- **Observation Space:** RGB image (flattened and normalized)
- **Frameworks:** PyTorch, Gymnasium

---

## Key Features

- Custom Actor-Critic neural network
- Learnable log standard deviation for continuous actions
- PPO clipping for stable policy updates
- GAE for advantage estimation
- GPU support if available
- TensorBoard logging for rewards and losses
