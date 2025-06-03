import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import pandas as pd

# Deterministic Iowa Gambling Task setup
class DeterministicIowaGamblingTask:
    def __init__(self):
        self.deck_cycles = {
            "A": [100, 100, -50, 100, -200, 100, -100, 100, -150, -250],
            "B": [100, 100, 100, 100, 100, 100, 100, 100, 100, -1150],
            "C": [50, 50, 25, 50, 0, 50, 0, 50, 0, -25],
            "D": [50, 50, 50, 50, 50, 50, 50, 50, 50, -200]
        }
        self.deck_indices = {deck: 0 for deck in self.deck_cycles}

    def draw_card(self, deck):
        idx = self.deck_indices[deck]
        reward = self.deck_cycles[deck][idx]
        self.deck_indices[deck] = (idx + 1) % 10
        deck_reset = self.deck_indices[deck] == 0
        return reward, deck_reset

# Agents
class DeterministicRLAgent:
    def __init__(self, alpha=0.05, epsilon=0.8, epsilon_decay=0.01, min_epsilon=0.1, epsilon_boost=1.05):
        self.values = {deck: 0.0 for deck in ["A", "B", "C", "D"]}
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_boost = epsilon_boost
        self.min_epsilon = min_epsilon

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(list(self.values.keys()))
        else:
            max_value = max(self.values.values())
            best_decks = [deck for deck, value in self.values.items() if value == max_value]
            return np.random.choice(best_decks)

    def update(self, deck, reward):
        self.values[deck] += self.alpha * (reward - self.values[deck])

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)

    def boost_epsilon(self):
        self.epsilon = min(1.0, self.epsilon * self.epsilon_boost)

class AmygdalaAgent(DeterministicRLAgent):
    def update(self, deck, reward):
        perceived = 0
        self.values[deck] += self.alpha * (perceived - self.values[deck])

class VMFAgent(DeterministicRLAgent):
    def __init__(self, **kwargs):
        super().__init__(alpha=0.0, **kwargs)

class GamblingDisorderAgent(DeterministicRLAgent):
    def __init__(self, reward_bias=2, loss_bias=0.2, **kwargs):
        super().__init__(alpha=0.5, **kwargs)
        self.reward_bias = reward_bias
        self.loss_bias = loss_bias

    def update(self, deck, reward):
        perceived = reward * self.reward_bias if reward >= 0 else reward * self.loss_bias
        self.values[deck] += self.alpha * (perceived - self.values[deck])

# Run simulation
def run_deterministic_simulation(agent_class, episodes=100):
    rewards_all, deck_counts_all, deck_choices_flat, ev_history_all = [], [], [], []
    for _ in range(100):  # 100 agents
        agent = agent_class()
        env = DeterministicIowaGamblingTask()
        rewards, deck_choices, ev_history = [], [], []
        for _ in range(episodes):
            deck = agent.select_action()
            reward, reset = env.draw_card(deck)
            agent.update(deck, reward)
            agent.decay_epsilon()
            if reset: agent.boost_epsilon()
            rewards.append(reward)
            deck_choices.append(deck)
            ev_history.append(agent.values.copy())
            deck_choices_flat.append(deck)
        rewards_all.append(rewards)
        deck_counts_all.append(Counter(deck_choices))
        ev_history_all.append(ev_history)
    return rewards_all, deck_counts_all, deck_choices_flat, ev_history_all

# Aggregate results
def aggregate_results(rewards_all, deck_counts_all):
    rewards_mean = np.mean(rewards_all, axis=0)
    deck_total = Counter()
    for d in deck_counts_all:
        deck_total.update(d)
    total_cards = len(rewards_all) * len(rewards_all[0])
    deck_percent = {k: v / total_cards * 100 for k, v in deck_total.items()}
    return rewards_mean, deck_percent

# Plot rewards + deck choice
def plot_all_results(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # Rewards subplot
    plt.figure(figsize=(10, 12))
    for i, (name, data) in enumerate(results.items(), 1):
        plt.subplot(4, 1, i)
        def moving_average(x, window=20):
            return np.convolve(x, np.ones(window)/window, mode='valid')

        avg_reward = np.mean(data["rewards"])
        smoothed = moving_average(data["rewards"], window=20)

        plt.plot(smoothed, label=f"{name} (20-trial MA)")
        plt.axhline(avg_reward, color='red', linestyle='--', linewidth=1.2, label=f"Mean: {avg_reward:.2f}")

        plt.title(f"{name} Agent - Smoothed Reward Over Time")
        plt.xlabel("Trial")
        plt.ylabel("Average Reward")
        plt.legend()
    plt.tight_layout()
    reward_path = os.path.join(save_dir, "reward_subplots_deterministic.png")
    plt.savefig(reward_path)
    plt.close()

    # Deck selection bar chart
    plt.figure(figsize=(10, 6))
    decks = ["A", "B", "C", "D"]
    bar_width = 0.2
    x = np.arange(len(decks))
    for i, (name, data) in enumerate(results.items()):
        pct = [data["deck_percent"].get(deck, 0) for deck in decks]
        bar = plt.bar(x + i * bar_width, pct, width=bar_width, label=name)
        for rect, val in zip(bar, pct):
            plt.text(rect.get_x() + rect.get_width() / 2., rect.get_height() + 0.5,
                     f"{val:.1f}%", ha='center', va='bottom', fontsize=8)
    plt.xticks(x + bar_width * 1.5, decks)
    plt.ylabel("Deck Choice Percentage")
    plt.title("Deck Selection (Deterministic, 100 Agents Each)")
    plt.legend()
    plt.tight_layout()
    deck_path = os.path.join(save_dir, "deck_choice_deterministic.png")
    plt.savefig(deck_path)
    plt.close()
    return reward_path, deck_path

# Plot Bechara-style subgraphs
def plot_advantage_disadvantage_subplots(deck_choices_all, save_dir):
    block_size = 20
    blocks = [f"{i+1}-{i+block_size}" for i in range(0, 100, block_size)]
    adv_decks = {'C', 'D'}
    disadv_decks = {'A', 'B'}

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axs = axs.flatten()

    for ax, (name, choices) in zip(axs, deck_choices_all.items()):
        adv, disadv = [], []
        for i in range(0, 100, block_size):
            block = choices[i * 100:(i + block_size) * 100]  # 100 agents × 20 trials
            adv.append(sum(d in adv_decks for d in block) / 100)
            disadv.append(sum(d in disadv_decks for d in block) / 100)

        ax.plot(blocks, disadv, marker='o', label="Disadvantageous (A&B)", color='black')
        ax.plot(blocks, adv, marker='o', label="Advantageous (C&D)", linestyle='--', color='gray')
        ax.set_title(f"{name} Agent")
        ax.set_ylim(0, 20)
        ax.set_xlabel("Trial Block")
        ax.set_ylabel("Avg Cards Selected")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    plt.suptitle("Card Selections from Advantageous vs Disadvantageous Decks", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    block_path = os.path.join(save_dir, "bechara_style_deck_block_plot.png")
    plt.savefig(block_path)
    plt.close()
    return block_path

def plot_ev_evolution(ev_histories, agent_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    decks = ['A', 'B', 'C', 'D']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)  # <- Explicitly False
    axes = axes.flatten()

    for idx, (agent_name, ev_history) in enumerate(zip(agent_names, ev_histories)):
        avg_ev = {deck: np.mean([[step[deck] for step in run] for run in ev_history], axis=0) for deck in decks}
        for deck in decks:
            axes[idx].plot(avg_ev[deck], label=f'Deck {deck}')
        axes[idx].set_title(agent_name)
        axes[idx].set_xlabel('Trial')  # Ensuring every subplot has Trial label
        axes[idx].set_ylabel('Estimated EV')  # Ensuring every subplot has EV label
        axes[idx].legend()

    plt.tight_layout()
    ev_plot_path = os.path.join(save_dir, "ev_evolution.png")
    plt.savefig(ev_plot_path)
    plt.close()
    return ev_plot_path


# Run and Save
save_dir = "igt_outputs_deterministic"
agent_classes = {
    "Healthy": DeterministicRLAgent,
    "Amygdala": AmygdalaAgent,
    "VMF": VMFAgent,
    "GamblingDisorder": GamblingDisorderAgent
}

results = {}
deck_choices_all = {}
ev_histories_all = []

for name, agent_class in agent_classes.items():
    rewards_all, deck_counts_all, flat_choices, ev_history = run_deterministic_simulation(agent_class)
    rewards_mean, deck_percent = aggregate_results(rewards_all, deck_counts_all)
    results[name] = {"rewards": rewards_mean, "deck_percent": deck_percent}
    deck_choices_all[name] = flat_choices
    ev_histories_all.append(ev_history)

reward_path, deck_path = plot_all_results(results, save_dir)
block_plot_path = plot_advantage_disadvantage_subplots(deck_choices_all, save_dir)
ev_plot_path = plot_ev_evolution(ev_histories_all, list(agent_classes.keys()), save_dir)

# Save CSV
rows = []
for name, data in results.items():
    row = {"Agent": name, "Mean_Reward": np.mean(data["rewards"])}
    row.update({f"Deck_{k}_Pct": v for k, v in data["deck_percent"].items()})
    rows.append(row)

df = pd.DataFrame(rows)
csv_path = os.path.join(save_dir, "aggregate_deterministic_results.csv")
df.to_csv(csv_path, index=False)

print("Saved:")
print(reward_path)
print(deck_path)
print(block_plot_path)
print(csv_path)
