# -*- coding: utf-8 -*-
"""
Visual Simulation for Forager's Gambit - Phase 6 

This script provides a visual representation of the co-evolutionary
simulation, where both creature and enemy populations evolve using
LSTM neural networks.
This version will split half the population between LStm and SNN to compare their performance in the same environment.


This version adds comprehensive data logging to a CSV file and a live,
dynamic Matplotlib graph to visualize population trends over generations.

Requires the Pygame and Matplotlib libraries:
pip install pygame matplotlib
"""

import pygame
import numpy as np
import random
import math
import csv
import matplotlib.pyplot as plt


# --- Constants & Colors ---
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
WORLD_PADDING = 20
APPLE_SIZE, HEAVY_APPLE_SIZE, CREATURE_SIZE = 10, 15, 8
SCORE_ZONE_SIZE, HIDING_SPOT_SIZE, ENEMY_SIZE = 50, 40, 10

COLOR_BACKGROUND = (30, 30, 30)
COLOR_CREATURE = (50, 205, 50)
COLOR_ENEMY = (200, 200, 200)
COLOR_APPLE = (50, 200, 50)
COLOR_HEAVY_APPLE = (255, 215, 0)
COLOR_SCORE_ZONE = (0, 0, 255, 100)
COLOR_HIDING_SPOT = (144, 238, 144, 100)
COLOR_TEXT = (255, 255, 255)
COLOR_LATCH_LINE = (150, 150, 150)


# --- Simulation Parameters ---
DEFAULT_CONFIG = {
    "creature_pop_size": 50,
    "enemy_pop_size": 4,
    "max_apples": 40,  
    "max_heavy_apples": 5,
    "apple_respawn_rate": 0.08,
    "starting_energy": 250.0, # Increased significantly
    "energy_decay_rate": 0.1,
    "gather_energy_bonus": 50.0, # Increased reward
    "heavy_apple_reward": 100.0,
    "max_energy": 350.0,
    "generation_time": 5000,
    "mutation_rate": 0.1,
    "mutation_strength": 0.5,
    "elitism_count": 2,
    "creature_hidden_size": 12,
    "enemy_hidden_size": 8,
    "tournament_size": 3,
    "enemy_attack_damage": 20.0,
    "enemy_fitness_bonus": 50.0
}


class LIF_Neuron:
    """A simple Leaky Integrate-and-Fire (LIF) neuron model."""
    def __init__(self, threshold=1.0, decay=0.9):
        self.threshold = threshold
        self.decay = decay
        self.potential = 0.0

    def receive_input(self, input_value):
        """Receives input and updates the neuron's potential."""
        self.potential += input_value
        self.potential *= self.decay

    def fire(self):
        """Determines if the neuron fires based on its potential."""
        if self.potential >= self.threshold:
            self.potential = 0.0 # Reset after firing
            return 1.0 # Spike output
        return 0.0 # No spike
    
class SNN_Layer:
    """A simple layer of spiking neurons."""
    def __init__(self, num_neurons, threshold=1.0, decay=0.9):
        self.neurons = [LIF_Neuron(threshold, decay) for _ in range(num_neurons)]

    def forward(self, inputs):
        """Processes inputs through the layer and returns spike outputs."""
        outputs = []
        for neuron in self.neurons:
            neuron.receive_input(np.sum(inputs)) # Simple summation of inputs
            outputs.append(neuron.fire())
        return np.array(outputs)
    
class SNN_NeuralNetwork:
    """A simple feedforward Spiking Neural Network."""
    def __init__(self, input_size, hidden_size, output_size):
        self.input_layer = SNN_Layer(input_size)
        self.hidden_layer = SNN_Layer(hidden_size)
        self.output_layer = SNN_Layer(output_size)

    def predict(self, inputs):
        """Performs a forward pass through the SNN."""
        hidden_outputs = self.hidden_layer.forward(self.input_layer.forward(inputs))
        output_spikes = self.output_layer.forward(hidden_outputs)
        return output_spikes
    
class LSTM_NeuralNetwork:
    """A simple NumPy-based LSTM Neural Network."""
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
         # --- LSTM Cell Weights ---
        # Each gate has weights for the input and the previous hidden state
        # We concatenate them for efficient matrix multiplication
        concat_size = input_size + hidden_size

        # --- LSTM Gate Weights & Biases ---
        self.wf, self.bf = np.random.randn(concat_size, hidden_size) * 0.1, np.zeros((1, hidden_size))
        self.wi, self.bi = np.random.randn(concat_size, hidden_size) * 0.1, np.zeros((1, hidden_size))
        
        self.wc, self.bc = np.random.randn(concat_size, hidden_size) * 0.1, np.zeros((1, hidden_size))
        self.wo, self.bo = np.random.randn(concat_size, hidden_size) * 0.1, np.zeros((1, hidden_size))
        
        # --- Output Layer ---
        self.w_out, self.b_out = np.random.randn(hidden_size, output_size) * 0.1, np.zeros((1, output_size))
        
        # --- LSTM State ---
        # These are unique to each creature and hold its short-term memory
        self.hidden_state, self.cell_state = np.zeros((1, hidden_size)), np.zeros((1, hidden_size))

    def reset_state(self):
        """Resets the hidden and cell states to zero."""
        self.hidden_state = np.zeros((1, self.hidden_size))
        self.cell_state = np.zeros((1, self.hidden_size))

    def predict(self, inputs):
        """Performs a forward pass through the LSTM."""
        inputs = np.array(inputs).reshape(1, -1)
        combined_input = np.concatenate((inputs, self.hidden_state), axis=1)

        f = 1 / (1 + np.exp(-(np.dot(combined_input, self.wf) + self.bf)))
        i = 1 / (1 + np.exp(-(np.dot(combined_input, self.wi) + self.bi)))
        c_candidate = np.tanh(np.dot(combined_input, self.wc) + self.bc)
        
        self.cell_state = f * self.cell_state + i * c_candidate
        
        o = 1 / (1 + np.exp(-(np.dot(combined_input, self.wo) + self.bo)))
        self.hidden_state = o * np.tanh(self.cell_state)
        
        output = np.tanh(np.dot(self.hidden_state, self.w_out) + self.b_out)
        return output[0]

    def get_weights(self):
        """Flattens all network weights into a single 1D array."""
        return np.concatenate([m.flatten() for m in [self.wf, self.bf, self.wi, self.bi, self.wc, self.bc, self.wo, self.bo, self.w_out, self.b_out]])

    def set_weights(self, flat_weights):
        """Sets network weights from a flattened 1D array."""
        s = 0
        matrices = [self.wf, self.bf, self.wi, self.bi, self.wc, self.bc, self.wo, self.bo, self.w_out, self.b_out]
        new_matrices = []
        for m in matrices:
            size = np.prod(m.shape)
            w = flat_weights[s : s + size].reshape(m.shape)
            new_matrices.append(w)
            s += size
        (self.wf, self.bf, self.wi, self.bi, self.wc, self.bc, 
         self.wo, self.bo, self.w_out, self.b_out) = new_matrices

    def mutate(self, rate, strength):
        """Applies random mutations to the network's weights."""
        weights = self.get_weights()
        mutation_mask = np.random.random(weights.shape) < rate
        mutations = np.random.randn(len(weights)) * strength
        weights[mutation_mask] += mutations[mutation_mask]
        self.set_weights(weights)

#change introduced in phase 3 to accomadate intelligent agents with memory and more complex decision making. This is the base class for both creatures and enemies, as they share movement and sensing logic.
class BaseAgent:
    """A base class for all agents in the simulation."""
    def __init__(self, x, y, nn, config):
        self.pos = np.array([x, y], dtype=np.float64)
        self.nn = nn
        self.config = config
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = 0.0
        self.max_speed = 3.0
        self.max_turn_rate = 0.1
        self.fitness = 0.0
        self.sight_radius = 150

    def get_nearest_object(self, objects, ignore_self=False):
        """Finds the nearest object in a list to the agent."""
        if ignore_self:
            objects = [o for o in objects if o is not self]

        if not objects:
            return None, self.sight_radius, 0.0

        positions = np.array([o.pos for o in objects])
        dist_sq = np.sum((positions - self.pos)**2, axis=1)
        min_idx = np.argmin(dist_sq)

        if dist_sq[min_idx] < self.sight_radius**2:
            dist = np.sqrt(dist_sq[min_idx])
            direction = positions[min_idx] - self.pos
            target_angle = math.atan2(direction[1], direction[0])
            relative_angle = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
            return objects[min_idx], dist, relative_angle / math.pi
        
        return None, self.sight_radius, 0.0

    def move(self, turn_request, accel_request):
        """Updates the agent's position and angle based on NN output."""
        self.angle = (self.angle + turn_request * self.max_turn_rate) % (2 * math.pi)
        
        if accel_request > 0:
            self.speed = min(self.speed + accel_request * 0.2, self.max_speed)
        else:
            self.speed *= 0.95 # Friction
        
        velocity = np.array([math.cos(self.angle), math.sin(self.angle)]) * self.speed
        self.pos += velocity
        self.pos[0] = np.clip(self.pos[0], WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING)
        self.pos[1] = np.clip(self.pos[1], WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING)

class Creature(BaseAgent):
    NUM_INPUTS, NUM_OUTPUTS = 15, 6
    def __init__(self, x, y, nn, config):
        super().__init__(x, y, nn, config)
        self.energy = self.config['starting_energy']
        self.max_energy = self.config['max_energy']
        self.apples_held = 0
        self.max_apples = 5
        self.apples_deposited_total = 0
        self.communication_signal = 0.0
        self.latch_count = 0.0
        self.is_latched = False
        self.latched_to = None
        self.color = (138, 43, 226) if isinstance(nn, SNN_NeuralNetwork) else COLOR_CREATURE

    def get_inputs(self, apples, heavy_apples, hiding_spots, score_zone, enemies, all_creatures):
        """Gathers sensory information for the neural network.""" 
        inputs = np.zeros(self.NUM_INPUTS)

        # Inputs 0-1: Nearest apple (distance and angle)
        _, d_a, a_a = self.get_nearest_object(apples)
        inputs[0:2] = [(self.sight_radius - d_a) / self.sight_radius, a_a]
        
        # Inputs 2-3: Score zone (distance and angle)
        z_d = score_zone.pos - self.pos
        d_z = np.linalg.norm(z_d)
        z_t_a = math.atan2(z_d[1], z_d[0])
        z_r_a = (z_t_a - self.angle + math.pi) % (2 * math.pi) - math.pi
        inputs[2:4] = [max(0, (SCREEN_WIDTH - d_z) / SCREEN_WIDTH), z_r_a / math.pi]

        #Inputs 4-5:Internal state: Energy level and apples held
        inputs[4:6] = [self.energy / self.max_energy, self.apples_held / self.max_apples]
        
        # Inputs 6-7: Nearest enemy (distance and angle)
        _, d_e, a_e = self.get_nearest_object(enemies)
        inputs[6:8] = [(self.sight_radius - d_e) / self.sight_radius, a_e]
        
        #Inputs 8-9: Nearest heavy apple (distance and angle)
        _, d_h, a_h = self.get_nearest_object(heavy_apples)
        inputs[8:10] = [(self.sight_radius - d_h) / self.sight_radius, a_h]
        
        # Inputs 10-11: Nearest hiding spot (distance and angle)
        _, d_hs, a_hs = self.get_nearest_object(hiding_spots)
        inputs[10:12] = [(self.sight_radius - d_hs) / self.sight_radius, a_hs]
        
        # Inputs 12-14: Nearest ally (distance, angle, communication signal)
        n_ally, d_ally, a_ally = self.get_nearest_object(all_creatures, ignore_self=True)
        inputs[12:15] = [(self.sight_radius - d_ally) / self.sight_radius, a_ally, n_ally.communication_signal if n_ally else 0.0]
        
        return inputs

    def update(self, apples, heavy_apples, hiding_spots, score_zone, enemies, all_creatures):
        """Main update logic for the creature for one simulation step.""" 
        inputs = self.get_inputs(apples, heavy_apples, hiding_spots, score_zone, enemies, all_creatures)
        turn, accel, _, deposit, comms, latch = self.nn.predict(inputs)
        self.communication_signal = comms

         # Update color based on communication signal
        if self.communication_signal > 0.5:
            self.color = (255, 69, 0) # OrangeRed
        elif self.communication_signal < -0.5:
            self.color = (0, 191, 255) # DeepSkyBlue
        else:
            self.color = COLOR_CREATURE

        # Latching logic
        if latch > 0.5 and not self.is_latched:
            n_h, d, _ = self.get_nearest_object(heavy_apples)
            if n_h and d < (CREATURE_SIZE + HEAVY_APPLE_SIZE):
                self.is_latched, self.latched_to = True, n_h
                n_h.current_lifters.append(self)
        elif latch < -0.5 and self.is_latched:
            if self in self.latched_to.current_lifters:
                 self.latched_to.current_lifters.remove(self)
            self.is_latched, self.latched_to = False, None

        
        # Movement logic with latching consideration
        if self.is_latched and self.latched_to.is_lifted():
            avg_a = sum(c.angle for c in self.latched_to.current_lifters) / len(self.latched_to.current_lifters)
            avg_s = sum(c.speed for c in self.latched_to.current_lifters) / len(self.latched_to.current_lifters)
            vel = np.array([math.cos(avg_a), math.sin(avg_a)]) * avg_s
            self.latched_to.pos = np.clip(self.latched_to.pos + vel, WORLD_PADDING, [SCREEN_WIDTH - WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING])
            for lifter in self.latched_to.current_lifters:
                lifter.pos = self.latched_to.pos
        else:
            self.move(turn, accel)

        #interactions with environment

        # Check for apple collection
        for apple in apples[:]:
            if np.linalg.norm(self.pos - apple.pos) < (CREATURE_SIZE + APPLE_SIZE):
                apples.remove(apple)
                if self.apples_held < self.max_apples:
                    self.fitness += 1 # Reward for collecting an apple
                    self.apples_held = min(self.apples_held + 1, self.max_apples)
                    
                self.energy = min(self.energy + self.config['gather_energy_bonus'], self.max_energy)
                break
        
        if deposit > 0.5 and np.linalg.norm(self.pos - score_zone.pos) < (SCORE_ZONE_SIZE + CREATURE_SIZE):
            if self.apples_held > 0:
                self.fitness += self.apples_held * 10
                self.apples_deposited_total += self.apples_held
                self.apples_held = 0
            if self.is_latched and self.latched_to and self.latched_to in heavy_apples and self.latched_to.is_lifted():
                rew = self.config['heavy_apple_reward'] / len(self.latched_to.current_lifters)
                lifters_copy = list(self.latched_to.current_lifters)
                for lifter in lifters_copy:
                    lifter.fitness += rew
                    lifter.apples_deposited_total += 1
                    lifter.is_latched = False
                    lifter.latched_to = None
                heavy_apples.remove(self.latched_to)

        # Energy decay
        self.energy -= self.config['energy_decay_rate']

        #check if latched heavy apple should be dropped due to energy depletion/ "death of the creature"
        if self.energy <= 0 and self.is_latched:
            if self in self.latched_to.current_lifters:
                self.latched_to.current_lifters.remove(self)
            self.is_latched = False


        return self.energy > 0

    def draw(self, screen, font):
        """Draws the creature, its energy bar, and apple count."""


        energy_ratio = max(0, self.energy / self.max_energy)
        energy_bar_width = int(CREATURE_SIZE * 2 * energy_ratio)
        energy_bar_pos = self.pos - np.array([CREATURE_SIZE, CREATURE_SIZE + 4])
        pygame.draw.rect(screen, (255, 0, 0), (*energy_bar_pos.astype(int), CREATURE_SIZE * 2, 3))
        pygame.draw.rect(screen, (0, 255, 0), (*energy_bar_pos.astype(int), energy_bar_width, 3))

        # creature body
        pygame.draw.circle(screen, self.color, self.pos.astype(int), CREATURE_SIZE)
        
        #direction line
        end_line = self.pos + np.array([math.cos(self.angle), math.sin(self.angle)]) * (CREATURE_SIZE + 5)
        pygame.draw.line(screen, (200, 200, 200), self.pos.astype(int), end_line.astype(int), 2)

        #self count for apples held
        if self.apples_held > 0:
            text = font.render(str(self.apples_held), True, COLOR_TEXT)
            screen.blit(text, (self.pos + np.array([-text.get_width() / 2, -CREATURE_SIZE - 20])).astype(int))

        

class Enemy(BaseAgent):
    NUM_INPUTS, NUM_OUTPUTS = 6, 3
    def __init__(self, x, y, nn, config):
        super().__init__(x, y, nn, config)
        self.max_speed = 2.5
        self.creatures_killed = 0

    def update(self, creatures, hiding_spots, all_enemies):
        vis_c = [c for c in creatures if not any(np.linalg.norm(c.pos - s.pos) < s.radius for s in hiding_spots)]
        
        inputs = np.zeros(self.NUM_INPUTS)
        _, d_p, a_p = self.get_nearest_object(vis_c)
        inputs[0:2] = [(self.sight_radius - d_p) / self.sight_radius, a_p]
        _, d_h, a_h = self.get_nearest_object(hiding_spots)
        inputs[2:4] = [(self.sight_radius - d_h) / self.sight_radius, a_h]
        _, d_ally, a_ally = self.get_nearest_object(all_enemies, ignore_self=True)
        inputs[4:6] = [(self.sight_radius - d_ally) / self.sight_radius, a_ally]

        turn, accel, attack = self.nn.predict(inputs)
        self.move(turn, accel)

        if attack > 0:
            for c in creatures:
                if np.linalg.norm(self.pos - c.pos) < (ENEMY_SIZE + CREATURE_SIZE):
                    c.energy -= self.config['enemy_attack_damage']
                    if c.energy <= 0:
                        self.fitness += self.config['enemy_fitness_bonus']
                        self.creatures_killed += 1
                    break
    
    def draw(self, screen):
        pygame.draw.circle(screen, COLOR_ENEMY, self.pos.astype(int), ENEMY_SIZE)
        end_pos = self.pos + np.array([math.cos(self.angle), math.sin(self.angle)]) * (ENEMY_SIZE + 3)
        pygame.draw.line(screen, (255, 255, 255), self.pos.astype(int), end_pos.astype(int), 2)
        temp_surface = pygame.Surface((self.sight_radius * 2, self.sight_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surface, (*COLOR_ENEMY, 40), (self.sight_radius, self.sight_radius), self.sight_radius, 1)
        screen.blit(temp_surface, (self.pos - self.sight_radius).astype(int))


# --- Environment Object Classes ---

class HeavyApple:
    def __init__(self, x, y): 
        self.pos = np.array([x,y])
        self.current_lifters = []   
        self.color = COLOR_HEAVY_APPLE
        self.is_lifted = False
        self.required_lifters = 2

    def draw(self, screen): 
        pygame.draw.circle(screen, self.color, self.pos.astype(int), HEAVY_APPLE_SIZE)

class Apple:
    def __init__(self, x, y): self.pos = np.array([x, y])
    def draw(self, screen): pygame.draw.circle(screen, COLOR_APPLE, self.pos.astype(int), APPLE_SIZE)

class ScoreZone:
    def __init__(self, x, y): 
        self.pos = np.array([x, y])
    def draw(self, screen):
        s = pygame.Surface((SCORE_ZONE_SIZE*2, SCORE_ZONE_SIZE*2), pygame.SRCALPHA)
        pygame.draw.circle(s, COLOR_SCORE_ZONE, (SCORE_ZONE_SIZE, SCORE_ZONE_SIZE), SCORE_ZONE_SIZE)
        screen.blit(s, (self.pos - SCORE_ZONE_SIZE).astype(int))

class HidingSpot:
    def __init__(self, x, y, radius): 
        self.pos, self.radius = np.array([x, y]), radius
    def draw(self, screen):
        s = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, COLOR_HIDING_SPOT, (self.radius, self.radius), self.radius)
        screen.blit(s, (self.pos - self.radius).astype(int))

# --- Genetic Algorithm ---
def reproduction(parents, config, pop_size, hidden_size, agent_class):
    next_pop = []
    if not parents: # Handle total extinction
        for _ in range(pop_size):
            nn_class = LSTM_NeuralNetwork if random.random() < 0.5 else SNN_NeuralNetwork
            nn = nn_class(agent_class.NUM_INPUTS, hidden_size, agent_class.NUM_OUTPUTS)
            agent = agent_class(random.uniform(WORLD_PADDING, SCREEN_WIDTH-WORLD_PADDING), random.uniform(WORLD_PADDING, SCREEN_HEIGHT-WORLD_PADDING), nn, config)
            next_pop.append(agent)
        return next_pop

    parents.sort(key=lambda c: c.fitness, reverse=True)

    for i in range(min(config['elitism_count'], len(parents))):
        nn_class = LSTM_NeuralNetwork if isinstance(parents[i].nn, LSTM_NeuralNetwork) else SNN_NeuralNetwork
        nn = nn_class(agent_class.NUM_INPUTS, hidden_size, agent_class.NUM_OUTPUTS)
        nn.set_weights(parents[i].nn.get_weights())
        agent = agent_class(random.uniform(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING), random.uniform(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING), nn, config)
        next_pop.append(agent)

    while len(next_pop) < pop_size:
        k = config['tournament_size']
        p1 = max(random.sample(parents, k=k) if len(parents) >= k else parents, key=lambda c: c.fitness)
        p2 = max(random.sample(parents, k=k) if len(parents) >= k else parents, key=lambda c: c.fitness)

        p1_w, p2_w = p1.nn.get_weights(), p2.nn.get_weights()
        crossover_point = len(p1_w) // 2
        child_w = np.concatenate([p1_w[:crossover_point], p2_w[crossover_point:]])
        
        nn_class = LSTM_NeuralNetwork if random.random() < 0.5 else SNN_NeuralNetwork
        nn = nn_class(agent_class.NUM_INPUTS, hidden_size, agent_class.NUM_OUTPUTS)
        nn.set_weights(child_w)
        nn.mutate(config['mutation_rate'], config['mutation_strength'])
        
        agent = agent_class(random.uniform(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING), random.uniform(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING), nn, config)
        next_pop.append(agent)

    return next_pop

# --- Analytics Functions ---

def setup_csv(filename="simulation_log.csv"):
    """Creates the CSV file and writes the header."""
    header = [
        "generation", "top_fitness", "avg_fitness", "num_survivors",
        "total_apples_scored", "top_enemy_fitness", "total_kills"
    ]
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    return filename

def log_generation_stats(filename, gen_data):
    """Appends a row of statistics for the completed generation."""
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(gen_data.values())

def setup_plot():
    """Initializes the Matplotlib plot."""
    plt.ion() # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    fig.tight_layout(pad=3.0)
    return fig, ax1, ax2

def update_plot(fig, ax1, ax2, history):
    """Clears and redraws the plot with the latest data."""
    # Plot 1: Creature Fitness
    ax1.clear()
    ax1.plot(history['gen'], history['top_fit'], label='Top Fitness', color='green')
    ax1.plot(history['gen'], history['avg_fit'], label='Avg Fitness', color='limegreen', linestyle='--')
    ax1.set_title('Creature Fitness Over Generations')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Population and Enemy Stats
    ax2.clear()
    ax2.plot(history['gen'], history['survivors'], label='Survivors', color='blue')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Creature Survivors', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax2b = ax2.twinx() # Create a second y-axis
    ax2b.clear() # Clear the second y-axis before plotting
    ax2b.plot(history['gen'], history['kills'], label='Total Kills', color='red', linestyle=':')
    ax2b.set_ylabel('Enemy Kills', color='red')
    ax2b.tick_params(axis='y', labelcolor='red')

    ax2.set_title('Population and Predator-Prey Dynamics')
    fig.canvas.draw()
    plt.pause(0.001)

# --- Main Simulation ---

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Project: Forager's Gambit - Co-Evolution Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    small_font = pygame.font.SysFont(None, 18)
    cfg = DEFAULT_CONFIG

    # --- Analytics Setup ---
    log_filename = setup_csv()
    fig, ax1, ax2 = setup_plot()
    history = {'gen': [], 'top_fit': [], 'avg_fit': [], 'survivors': [], 'kills': []}

    # Initialize environment
    score_zone = ScoreZone(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
    hiding_spots = [HidingSpot(100, 100, HIDING_SPOT_SIZE), HidingSpot(700, 500, HIDING_SPOT_SIZE)]
    
    # Initialize populations
    creatures = [Creature(random.uniform(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING), random.uniform(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING), LSTM_NeuralNetwork(Creature.NUM_INPUTS, cfg['creature_hidden_size'], Creature.NUM_OUTPUTS), cfg) for _ in range(cfg['creature_pop_size'] // 2)]
    enemies = [Enemy(random.uniform(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING), random.uniform(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING), LSTM_NeuralNetwork(Enemy.NUM_INPUTS, cfg['enemy_hidden_size'], Enemy.NUM_OUTPUTS), cfg) for _ in range(cfg['enemy_pop_size'])]
    creatures += [Creature(random.uniform(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING), random.uniform(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING), SNN_NeuralNetwork(Creature.NUM_INPUTS, cfg['creature_hidden_size'], Creature.NUM_OUTPUTS), cfg) for _ in range(cfg['creature_pop_size'] // 2)]
    
    # --- LOGIC CHANGE: Use a generation_pool for reproduction ---
    generation_pool = list(creatures)
    enemy_pool = list(enemies)
    
    apples, heavy_apples = [], []
    
    gen, frame, running = 1, 0, True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Apple Spawning Logic
        if len(apples) < cfg['max_apples'] and random.random() < cfg['apple_respawn_rate']:
            apples.append(Apple(random.uniform(WORLD_PADDING, SCREEN_WIDTH-WORLD_PADDING), random.uniform(WORLD_PADDING, SCREEN_HEIGHT-WORLD_PADDING)))
        
        # Agent Update Logic
        if creatures:
            for i in range(len(creatures) - 1, -1, -1):
                if not creatures[i].update(apples, heavy_apples, hiding_spots, score_zone, enemies, creatures):
                    creatures.pop(i)
            
            for enemy in enemies:
                enemy.update(creatures, hiding_spots, enemies)
        
        frame += 1
        
        # --- Generation Reset ---
        if frame >= cfg['generation_time'] or not creatures:
            # --- Data Collection for Analytics ---
            num_survivors = len(creatures)
            
            if generation_pool:
                top_fitness = max(c.fitness for c in generation_pool)
                avg_fitness = sum(c.fitness for c in generation_pool) / len(generation_pool)
                total_apples = sum(c.apples_deposited_total for c in generation_pool)
            else:
                top_fitness, avg_fitness, total_apples = 0, 0, 0
            
            if enemy_pool:
                top_enemy_fitness = max(e.fitness for e in enemy_pool) if enemy_pool else 0
                total_kills = sum(e.creatures_killed for e in enemy_pool)
            else:
                top_enemy_fitness, total_kills = 0, 0
                
            # Log data to CSV
            stats = {
                "generation": gen, "top_fitness": top_fitness, "avg_fitness": avg_fitness,
                "num_survivors": num_survivors, "total_apples_scored": total_apples,
                "top_enemy_fitness": top_enemy_fitness, "total_kills": total_kills
            }
            log_generation_stats(log_filename, stats)

            # Update history for plotting
            history['gen'].append(gen)
            history['top_fit'].append(top_fitness)
            history['avg_fit'].append(avg_fitness)
            history['survivors'].append(num_survivors)
            history['kills'].append(total_kills)
            
            # Update plot every 5 generations to reduce slowdown
            if gen % 5 == 0:
                update_plot(fig, ax1, ax2, history)

            print(f"Generation {gen} finished. Top fitness: {top_fitness:.2f}, Survivors: {num_survivors}")
            gen += 1
            frame = 0
            
            # --- Reproduction using the full generation pool ---
            creatures = reproduction(generation_pool, cfg, cfg['creature_pop_size'], cfg['creature_hidden_size'], Creature)
            generation_pool = list(creatures) # Reset the pool
            
            enemies = reproduction(enemy_pool, cfg, cfg['enemy_pop_size'], cfg['enemy_hidden_size'], Enemy)
            enemy_pool = list(enemies)

            apples.clear()
            heavy_apples.clear()
            for agent in creatures + enemies:
                agent.nn.reset_state()

        # --- Drawing Logic ---
        screen.fill(COLOR_BACKGROUND)
        
        score_zone.draw(screen)
        for spot in hiding_spots: 
            spot.draw(screen)
        for apple in apples: 
            apple.draw(screen)
        
        for creature in creatures: 
            creature.draw(screen, small_font)
        for enemy in enemies: 
            enemy.draw(screen)
            
        # UI Text
        screen.blit(font.render(f"Generation: {gen}", True, COLOR_TEXT), (10, 10))
        screen.blit(font.render(f"Time: {frame}/{cfg['generation_time']}", True, COLOR_TEXT), (10, 30))
        screen.blit(font.render(f"Creatures: {len(creatures)}/{cfg['creature_pop_size']}", True, COLOR_TEXT), (10, 50))
        #specific creature type counts
        snn_count = sum(1 for c in creatures if isinstance(c.nn, SNN_NeuralNetwork))
        lstm_count = sum(1 for c in creatures if isinstance(c.nn, LSTM_NeuralNetwork))
        screen.blit(font.render(f"SNN: {snn_count} | LSTM: {lstm_count}", True, COLOR_TEXT), (10, 70))
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    plt.ioff() # Turn off interactive mode
    plt.show() # Keep the final plot window open

if __name__ == '__main__':
    main()
