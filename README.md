# 🧠 Emotional Regulation and Decision-Making: A Reinforcement Learning Simulation

This project explores how **emotional regulation shapes decision-making** through the lens of computational modeling. By simulating cognitive-affective impairments using reinforcement learning (RL), we replicate decision patterns observed in lesion and addiction studies—offering strong support for the **Somatic Marker Hypothesis**.

> 📝 Based on the paper:  
> _“The Role of Emotional Regulation in Decision-Making: A Balance Between Cognition and Emotion”_  
> 🧑‍🎓 Author: Agnibho Chattarji  
> 🏛 Georgia Institute of Technology  
> 📄 [Read Full Paper](Emotional_Regulation_and_Regular_Decision_Making_Final.pdf)

---

## 🔬 Core Hypothesis

Decision-making is not merely rational or emotional—it is a **dynamic interplay** of both systems. Balanced emotional regulation enables individuals to adaptively weigh rewards and punishments over time. Disruptions—either hypo- or hyperactivation—can severely impair decision-making.

---

## 🧪 Agent Profiles Simulated

| Agent Type         | Cognitive-Affective Profile                   | Learning Dynamics                                       |
|--------------------|-----------------------------------------------|----------------------------------------------------------|
| **Healthy**        | Balanced emotion and cognition                | Gradual learning; avoids risky decks over time           |
| **Amygdala-Damaged** | Lacks anticipatory emotion                   | Perceives no reward signal; fails to form preferences    |
| **VMF-Damaged**    | No integration of emotional feedback          | Cannot learn from outcomes (α = 0.0)                     |
| **Gambling Disorder** | Emotionally dysregulated, reward-biased     | Overreacts to gains, ignores losses (α = 0.5, bias applied) |

<img src="igt_outputs_deterministic\Agent.png" width="700"/>

---

## 🧠 Simulation Environment: The Iowa Gambling Task (IGT)

This environment mimics a **deterministic IGT** with 4 decks:

- **A, B**: High short-term rewards, net long-term loss  
- **C, D**: Lower immediate gain, but net positive

Agents use a **Q-learning** architecture with ε-greedy exploration.  
Behavior is shaped by tuning parameters like:
- Learning rate `α`
- Exploration rate `ε`
- Reward asymmetry (for biased agents)

---

## 📊 Key Results

<img src="igt_outputs_deterministic\ev_evolution.png" width="1000"/>
*Healthy agents gradually learn to avoid bad decks; Gambling agents persist in risky behavior.*

<img src="igt_outputs_deterministic\deck_choice_deterministic.png" width="1000"/>
*Deck preferences highlight agent-specific cognitive-affective impairments.*

---


