# 🧠 Emotional Regulation and Decision-Making: A Reinforcement Learning Simulation

This project investigates how emotional regulation impacts human decision-making by simulating cognitive-affective profiles using reinforcement learning (RL). It models healthy and impaired agents (Amygdala-lesioned, VMF-lesioned, and Gambling Disorder profiles) on a simulated Iowa Gambling Task (IGT).

<img src="igt_outputs_deterministic\Agent.png" width="700"/>

## 🧩 Core Hypothesis

Decision-making is optimized not by purely rational or purely emotional mechanisms, but through a dynamic balance of both. This aligns with the **Somatic Marker Hypothesis**, which proposes that affective signals guide cognition.

---

## 🧪 Agents Simulated

| Agent Type       | Emotional Profile                           | Learning Traits                         |
|------------------|----------------------------------------------|------------------------------------------|
| Healthy          | Balanced affect-cognition                   | Moderate learning, exploration decay     |
| Amygdala-Damaged | Absent anticipatory emotion                 | Zero perceived reward, flat behavior     |
| VMF-Damaged      | Cognitive-emotional disintegration          | No learning over time                    |
| Gambling Disorder| Hyperactive reward salience, loss blunting | Inflated $\alpha$, reward-loss bias      |

---

## 📊 Key Results

<img src="igt_outputs_deterministic\ev_evolution.png" width="700"/>
*Healthy agents gradually learn to avoid bad decks; Gambling agents persist in risky behavior.*

<img src="igt_outputs_deterministic\deck_choice_deterministic.png" width="700"/>
*Deck preferences highlight agent-specific cognitive-affective impairments.*

---

## 📁 Repository Structure

