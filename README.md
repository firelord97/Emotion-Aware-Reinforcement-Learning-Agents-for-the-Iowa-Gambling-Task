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

- The **Somatic Marker Hypothesis** \parencite{Damasio1996, Bechara1999} posits that emotional signals—“somatic markers”—guide future-oriented decision-making under uncertainty.
- **Amygdala and VMF lesions**, as shown in Bechara et al. (1999), eliminate emotional feedback or impair its cognitive integration, resulting in disorganized or nonadaptive choices.
- **Gambling disorder** studies \parencite{Clark2010, Mari2024} reveal the consequences of excessive emotional salience—e.g., overvaluing wins and ignoring losses—on compulsive decision-making.
- **Reinforcement Learning (RL)** provides a computational lens for simulating how different affective-cognitive profiles influence behavior over time, with tunable parameters like learning rate and reward asymmetry.

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

<img src="igt_outputs_deterministic\ev_evolution.png" width="900"/>

*Healthy agents gradually learn to avoid bad decks; Gambling agents persist in risky behavior.*

<img src="igt_outputs_deterministic\deck_choice_deterministic.png" width="900"/>

*Deck preferences highlight agent-specific cognitive-affective impairments.*

---


---

## 📌 Key Takeaways

- Balanced emotional regulation is essential for optimal decision-making.
- RL agents can model both healthy and impaired human behavior.
- The **Somatic Marker Hypothesis** gains strong computational support.
- Emotionally-informed RL can inform clinical understanding of impulsivity, addiction, and emotional dysregulation.

---

## 🔮 Future Work

- Simulate **bounded memory RL agents** to model “gut-feeling” based heuristics
- Test how tuning `α`, `ε`, and reward asymmetries affect learning across agents
- Map RL parameter dynamics to human traits like impulsivity or emotional blunting
- Expand to stochastic or real-time decision-making tasks

---

## 📚 References

Key literature includes:
- Bechara et al. (1999), Iowa Gambling Task
- Damasio (1996), Somatic Marker Hypothesis
- Clark (2010), cognitive distortions in gambling
- Gershman (2015), RL in human behavior modeling

---
