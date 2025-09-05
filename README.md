# text2motion-laparoscopy
Text-to-Motion for Laparoscopic Surgery

This repository contains the code, dataset, and evaluation notebooks for the MSc Robotics dissertation project:
“A Text-to-Motion System for Robotic Control of Endoscopic Camera in Laparoscopic Surgery” (University of Birmingham, 2025).

Project Overview

We developed a natural language (NL) → motion control system for laparoscopic camera positioning using a UR5e robot.

Fine-tuned Qwen2.5-7B with LoRA adapters.

Optimized hyperparameters (learning rate, rank, dropout) using Optuna.

Dataset built with ChArUco-based pose mapping.


Key Results

97.60% exact-match accuracy on held-out test set (n=750).

97.77% accuracy in human evaluation (3 users × 9 commands × 10 variants).

Mean latency: 0.618 s from NL input to parsed command.
