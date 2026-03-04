import numpy as np
import random
import math
import os
from datetime import datetime

# --- Constants ---
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
WORLD_PADDING = 20
APPLE_SIZE, HEAVY_APPLE_SIZE, CREATURE_SIZE = 10, 15, 8
SCORE_ZONE_SIZE, HIDING_SPOT_SIZE, ENEMY_SIZE = 50, 40, 10

# --- Default Configuration ---
DEFAULT_CONFIG = {
    "creature_pop_size": 50,
    "enemy_pop_size": 4,
    "max_apples": 20,
    "max_heavy_apples": 5,
    "apple_respawn_rate": 0.07,
    "starting_energy": 120.0,
    "energy_decay_rate": 0.1,
    "gather_energy_bonus": 30.0,
    "heavy_apple_reward": 60.0,
    "max_energy": 200.0,
    "generation_time": 5000,
    "mutation_rate": 0.1,
    "mutation_strength": 0.5,
    "elitism_count": 2,
    "creature_hidden_size": 12,
    "enemy_hidden_size": 8,
    "tournament_size": 3,
    "enemy_attack_damage": 5.0,
    "enemy_fitness_bonus": 50.0
}

class LSTM_NeuralNetwork:
    """LSTM Neural Network implementation for headless simulation"""
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize LSTM weights
        self.hidden_size = hidden_size
        
        # Input gate
        self.Wii = np.random.randn(hidden_size, input_size) * 0.01
        self.Whi = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bi = np.zeros((hidden_size, 1))
        
        # Forget gate
        self.Wif = np.random.randn(hidden_size, input_size) * 0.01
        self.Whf = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))
        
        # Cell gate
        self.Wig = np.random.randn(hidden_size, input_size) * 0.01
        self.Whg = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bg = np.zeros((hidden_size, 1))
        
        # Output gate
        self.Wio = np.random.randn(hidden_size, input_size) * 0.01
        self.Who = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bo = np.zeros((hidden_size, 1))
        
        # Output layer
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.by = np.zeros((output_size, 1))
        
        self.reset_state()

    def reset_state(self):
        """Reset the LSTM state (memory cell and hidden state)"""
        self.h = np.zeros((self.hidden_size, 1))
        self.c = np.zeros((self.hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def predict(self, inputs):
        """Forward pass through the LSTM network"""
        x = np.array(inputs).reshape(-1, 1)
        
        # Input gate
        i = self.sigmoid(np.dot(self.Wii, x) + np.dot(self.Whi, self.h) + self.bi)
        
        # Forget gate
        f = self.sigmoid(np.dot(self.Wif, x) + np.dot(self.Whf, self.h) + self.bf)
        
        # Cell gate
        g = np.tanh(np.dot(self.Wig, x) + np.dot(self.Whg, self.h) + self.bg)
        
        # Output gate
        o = self.sigmoid(np.dot(self.Wio, x) + np.dot(self.Who, self.h) + self.bo)
        
        # Memory cell and hidden state
        self.c = f * self.c + i * g
        self.h = o * np.tanh(self.c)
        
        # Output layer
        y = np.dot(self.Why, self.h) + self.by
        return np.tanh(y).flatten()

    def get_weights(self):
        """Get all weights as a flat array"""
        return np.concatenate([
            self.Wii.flatten(), self.Whi.flatten(), self.bi.flatten(),
            self.Wif.flatten(), self.Whf.flatten(), self.bf.flatten(),
            self.Wig.flatten(), self.Whg.flatten(), self.bg.flatten(),
            self.Wio.flatten(), self.Who.flatten(), self.bo.flatten(),
            self.Why.flatten(), self.by.flatten()
        ])

    def set_weights(self, weights):
        """Set weights from a flat array"""
        idx = 0
        
        def get_slice(shape):
            nonlocal idx
            size = np.prod(shape)
            result = weights[idx:idx + size]
            idx += size
            return result.reshape(shape)

        self.Wii = get_slice(self.Wii.shape)
        self.Whi = get_slice(self.Whi.shape)
        self.bi = get_slice(self.bi.shape)
        
        self.Wif = get_slice(self.Wif.shape)
        self.Whf = get_slice(self.Whf.shape)
        self.bf = get_slice(self.bf.shape)
        
        self.Wig = get_slice(self.Wig.shape)
        self.Whg = get_slice(self.Whg.shape)
        self.bg = get_slice(self.bg.shape)
        
        self.Wio = get_slice(self.Wio.shape)
        self.Who = get_slice(self.Who.shape)
        self.bo = get_slice(self.bo.shape)
        
        self.Why = get_slice(self.Why.shape)
        self.by = get_slice(self.by.shape)


    def mutate(self, rate, strength):
        """Mutate the network weights"""
        weights = self.get_weights()
        mask = np.random.random(weights.shape) < rate
        mutation = np.random.randn(mask.sum()) * strength
        weights[mask] += mutation
        self.set_weights(weights)

class BaseAgent:
    """Base class for both creatures and enemies"""
    def __init__(self, x, y, nn, config):
        self.pos = np.array([x, y], dtype=np.float64)
        self.velocity = np.array([0.0, 0.0], dtype=np.float64)
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = 0.0
        self.nn = nn
        self.config = config
        self.fitness = 0.0

    def get_nearest_object(self, objects):
        if not objects:
            return None, float('inf')

        my_pos = self.pos.reshape(1, 2)
        object_positions = np.array([obj.pos for obj in objects])
        
        distances = np.linalg.norm(object_positions - my_pos, axis=1)
        
        nearest_index = np.argmin(distances)
        min_dist = distances[nearest_index]
        
        return objects[nearest_index], min_dist


    def move(self, turn_request, accel_request):
        # Update angle
        self.angle += turn_request * 0.1
        self.angle %= (2 * math.pi)
        
        # Update speed
        self.speed += accel_request * 0.1
        self.speed = max(0, min(self.speed, 3.0))
        
        # Update position
        direction = np.array([math.cos(self.angle), math.sin(self.angle)])
        self.velocity = direction * self.speed
        new_pos = self.pos + self.velocity
        
        # Boundary checking
        new_pos[0] = np.clip(new_pos[0], WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING)
        new_pos[1] = np.clip(new_pos[1], WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING)
        
        self.pos = new_pos

class Creature(BaseAgent):
    """Creature class with LSTM brain"""
    NUM_INPUTS = 12 
    NUM_OUTPUTS = 5  # [turn, accelerate, gather, communicate, hide]
    
    def __init__(self, x, y, nn, config):
        super().__init__(x, y, nn, config)
        self.energy = config['starting_energy']
        self.apples_held = 0
        self.max_apples = 3
        self.communication_signal = 0.0
        self.is_hiding = False
        self.lifetime = 0
        self.apples_deposited = 0

    def update(self, apples, heavy_apples, hiding_spots, score_zone, enemies, all_creatures):
        if self.energy <= 0:
            return False

        # Get neural network inputs
        inputs = self.get_inputs(apples, heavy_apples, hiding_spots, score_zone, enemies, all_creatures)
        outputs = self.nn.predict(inputs)
        
        # Parse outputs
        turn, accel, gather, self.communication_signal, hide_request = outputs
        
        # ### --- FIX 1: REORDERED LOGIC & INCREASED REACH --- ###
        # Perform actions based on current position BEFORE moving for the next frame.

        # 1. Gathering Action
        if gather > 0 and self.apples_held < self.max_apples:
            all_apples = apples + heavy_apples
            nearest_apple, dist = self.get_nearest_object(all_apples)
            
            # Make the creature's "reach" larger for easier gathering
            effective_reach = CREATURE_SIZE * 1.5
            if nearest_apple:
                apple_radius = HEAVY_APPLE_SIZE if isinstance(nearest_apple, HeavyApple) else APPLE_SIZE
                if dist < (effective_reach + apple_radius):
                    if isinstance(nearest_apple, HeavyApple):
                        self.energy += self.config['heavy_apple_reward']
                        heavy_apples.remove(nearest_apple)
                    else:
                        self.energy += self.config['gather_energy_bonus']
                        apples.remove(nearest_apple)
                    self.apples_held += 1
                    # Give fitness for the simple act of gathering
                    self.fitness += 1
                    self.energy = min(self.energy, self.config['max_energy'])

        # 2. Depositing Action
        dist_to_score = np.linalg.norm(self.pos - score_zone.pos)
        if self.apples_held > 0 and dist_to_score < SCORE_ZONE_SIZE:
            self.fitness += self.apples_held * 10
            self.apples_deposited += self.apples_held
            self.apples_held = 0

        # 3. Hiding Action
        if hide_request > 0:
            nearest_spot, dist = self.get_nearest_object(hiding_spots)
            self.is_hiding = nearest_spot and dist < HIDING_SPOT_SIZE
        else:
            self.is_hiding = False
        
        # 4. Movement Action (now happens last)
        self.move(turn, accel)
        
        # 5. Final state updates
        self.energy -= self.config['energy_decay_rate']
        self.lifetime += 1
        return True


    def get_inputs(self, apples, heavy_apples, hiding_spots, score_zone, enemies, all_creatures):
        
        def get_relative_info(target_object):
            if target_object:
                dist = np.linalg.norm(target_object.pos - self.pos)
                angle = math.atan2(target_object.pos[1] - self.pos[1], target_object.pos[0] - self.pos[0])
                return [dist / SCREEN_WIDTH, angle / math.pi]
            return [1.0, 0.0]

        # Nearest apple
        nearest_apple, _ = self.get_nearest_object(apples + heavy_apples)
        inputs = get_relative_info(nearest_apple)

        # Score zone
        inputs.extend(get_relative_info(score_zone))

        # Nearest enemy
        nearest_enemy, _ = self.get_nearest_object(enemies)
        inputs.extend(get_relative_info(nearest_enemy))

        # Nearest hiding spot
        nearest_spot, _ = self.get_nearest_object(hiding_spots)
        inputs.extend(get_relative_info(nearest_spot))

        # Internal state
        inputs.extend([
            self.energy / self.config['max_energy'],
            self.apples_held / self.max_apples,
            self.speed / 3.0,
            float(self.is_hiding)
        ])

        return inputs

class Enemy(BaseAgent):
    """Enemy class with LSTM brain"""
    NUM_INPUTS = 8 
    NUM_OUTPUTS = 3  # [turn, accelerate, attack]
    
    def __init__(self, x, y, nn, config):
        super().__init__(x, y, nn, config)
        self.attack_cooldown = 0
        self.creatures_caught = 0

    def update(self, creatures, hiding_spots, all_enemies):
        inputs = self.get_inputs(creatures, hiding_spots, all_enemies)
        outputs = self.nn.predict(inputs)
        
        # Movement
        self.move(outputs[0], outputs[1])
        
        # Attack
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        
        if outputs[2] > 0 and self.attack_cooldown == 0:
            for creature in creatures:
                if not creature.is_hiding and np.linalg.norm(creature.pos - self.pos) < (ENEMY_SIZE + CREATURE_SIZE):
                    creature.energy -= self.config['enemy_attack_damage']
                    if creature.energy <= 0:
                        self.fitness += self.config['enemy_fitness_bonus']
                        self.creatures_caught += 1
                    self.attack_cooldown = 20
                    break
        
        return True

    def get_inputs(self, creatures, hiding_spots, all_enemies):
        
        def get_relative_info(target_object):
            if target_object:
                dist = np.linalg.norm(target_object.pos - self.pos)
                angle = math.atan2(target_object.pos[1] - self.pos[1], target_object.pos[0] - self.pos[0])
                return [dist / SCREEN_WIDTH, angle / math.pi]
            return [1.0, 0.0]

        # Nearest visible creature (not hiding)
        visible_creatures = [c for c in creatures if not c.is_hiding]
        nearest_creature, _ = self.get_nearest_object(visible_creatures)
        inputs = get_relative_info(nearest_creature)

        # Nearest hiding spot
        nearest_spot, _ = self.get_nearest_object(hiding_spots)
        inputs.extend(get_relative_info(nearest_spot))

        # Internal state
        inputs.extend([
            self.speed / 3.0,
            self.attack_cooldown / 20.0,
            len(visible_creatures) / (len(creatures) + 1e-6),
            self.creatures_caught / 10.0
        ])
        
        return inputs

class Apple:
    def __init__(self, x, y):
        self.pos = np.array([x, y])

class HeavyApple:
    def __init__(self, x, y):
        self.pos = np.array([x, y])

class ScoreZone:
    def __init__(self, x, y):
        self.pos = np.array([x, y])

class HidingSpot:
    def __init__(self, x, y):
        self.pos = np.array([x, y])

def setup_csv_logger(filename):
    header = [
        'generation', 'creatures_survived', 'enemies_survived',
        'best_creature_fitness', 'avg_creature_fitness',
        'best_enemy_fitness', 'avg_enemy_fitness',
        'total_apples_deposited', 'total_creatures_caught'
    ]
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='') as f:
        f.write(','.join(header) + '\n')

def log_generation_stats(filename, stats):
    with open(filename, mode='a', newline='') as f:
        header = [
            'generation', 'creatures_survived', 'enemies_survived',
            'best_creature_fitness', 'avg_creature_fitness',
            'best_enemy_fitness', 'avg_enemy_fitness',
            'total_apples_deposited', 'total_creatures_caught'
        ]
        row = [str(stats[key]) for key in header]
        f.write(','.join(row) + '\n')


def reproduction(parents, config, pop_size, hidden_size, agent_class):
    next_pop = []
    if not parents: 
        for _ in range(pop_size):
            nn = LSTM_NeuralNetwork(agent_class.NUM_INPUTS, hidden_size, agent_class.NUM_OUTPUTS)
            agent = agent_class(random.uniform(WORLD_PADDING, SCREEN_WIDTH-WORLD_PADDING), random.uniform(WORLD_PADDING, SCREEN_HEIGHT-WORLD_PADDING), nn, config)
            next_pop.append(agent)
        return next_pop

    parents.sort(key=lambda c: c.fitness, reverse=True)

    for i in range(min(config['elitism_count'], len(parents))):
        nn = LSTM_NeuralNetwork(agent_class.NUM_INPUTS, hidden_size, agent_class.NUM_OUTPUTS)
        nn.set_weights(parents[i].nn.get_weights())
        agent = agent_class(random.uniform(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING), random.uniform(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING), nn, config)
        next_pop.append(agent)

    while len(next_pop) < pop_size:
        k = config['tournament_size']
        # Handle cases where population is smaller than tournament size
        p1 = max(random.sample(parents, k=k) if len(parents) >= k else parents, key=lambda c: c.fitness)
        p2 = max(random.sample(parents, k=k) if len(parents) >= k else parents, key=lambda c: c.fitness)

        p1_w, p2_w = p1.nn.get_weights(), p2.nn.get_weights()
        crossover_point = len(p1_w) // 2
        child_w = np.concatenate([p1_w[:crossover_point], p2_w[crossover_point:]])
        
        nn = LSTM_NeuralNetwork(agent_class.NUM_INPUTS, hidden_size, agent_class.NUM_OUTPUTS)
        nn.set_weights(child_w)
        nn.mutate(config['mutation_rate'], config['mutation_strength'])
        
        agent = agent_class(random.uniform(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING), random.uniform(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING), nn, config)
        next_pop.append(agent)

    return next_pop

def run_headless_simulation(config={}, total_generations=100, results_csv_path='results.csv'):
    """Main simulation loop"""
    cfg = {**DEFAULT_CONFIG, **config}
    setup_csv_logger(results_csv_path)
    
    # Initialize environment
    score_zone = ScoreZone(SCREEN_WIDTH/2, SCREEN_HEIGHT/2)
    hiding_spots = [
        HidingSpot(SCREEN_WIDTH/4, SCREEN_HEIGHT/4),
        HidingSpot(3*SCREEN_WIDTH/4, SCREEN_HEIGHT/4),
        HidingSpot(SCREEN_WIDTH/4, 3*SCREEN_HEIGHT/4),
        HidingSpot(3*SCREEN_WIDTH/4, 3*SCREEN_HEIGHT/4)
    ]
    
    # Initialize populations
    creatures = [Creature(
        random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
        random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING),
        LSTM_NeuralNetwork(Creature.NUM_INPUTS, cfg['creature_hidden_size'], Creature.NUM_OUTPUTS),
        cfg
    ) for _ in range(cfg['creature_pop_size'])]
    
    enemies = [Enemy(
        random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
        random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING),
        LSTM_NeuralNetwork(Enemy.NUM_INPUTS, cfg['enemy_hidden_size'], Enemy.NUM_OUTPUTS),
        cfg
    ) for _ in range(cfg['enemy_pop_size'])]

    for generation in range(1, total_generations + 1):
        # ### --- FIX 2: IMPLEMENT GENERATION POOL --- ###
        # This pool keeps all agents from the start of the generation for reproduction,
        # ensuring the genes of high-scorers who die are not lost.
        creature_generation_pool = list(creatures)
        enemy_generation_pool = list(enemies)

        # Reset environment
        apples = [Apple(
            random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
            random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING)
        ) for _ in range(cfg['max_apples'])]
        
        heavy_apples = [HeavyApple(
            random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
            random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING)
        ) for _ in range(cfg['max_heavy_apples'])]

        # Reset agents' states for the new generation
        for agent_pop in [creatures, enemies]:
            for agent in agent_pop:
                agent.pos = np.array([
                    random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
                    random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING)
                ])
                agent.fitness = 0
                if isinstance(agent, Creature):
                    agent.energy = cfg['starting_energy']
                    agent.apples_held = 0
                    agent.apples_deposited = 0
                    agent.lifetime = 0
                else: # Enemy
                    agent.creatures_caught = 0
                agent.nn.reset_state()


        # Simulation loop
        for _ in range(cfg['generation_time']):
            # Update creatures (loop backwards for safe removal)
            for i in range(len(creatures) - 1, -1, -1):
                c = creatures[i]
                if not c.update(apples, heavy_apples, hiding_spots, score_zone, enemies, creatures):
                    creatures.pop(i) # Remove from active list, but it remains in the pool
            
            # If all creatures die, end the generation early
            if not creatures:
                break

            # Update enemies
            for e in enemies:
                e.update(creatures, hiding_spots, enemies)

            # Spawn new apples
            if random.random() < cfg['apple_respawn_rate']:
                if len(apples) < cfg['max_apples']:
                    apples.append(Apple(
                        random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
                        random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING)
                    ))

                if len(heavy_apples) < cfg['max_heavy_apples']:
                    heavy_apples.append(HeavyApple(
                        random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
                        random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING)
                    ))

        # Calculate statistics using the full generation pools
        stats = {
            'generation': generation,
            'creatures_survived': len(creatures),
            'enemies_survived': len(enemies),
            'best_creature_fitness': max([c.fitness for c in creature_generation_pool], default=0),
            'avg_creature_fitness': np.mean([c.fitness for c in creature_generation_pool]) if creature_generation_pool else 0,
            'best_enemy_fitness': max([e.fitness for e in enemy_generation_pool], default=0),
            'avg_enemy_fitness': np.mean([e.fitness for e in enemy_generation_pool]) if enemy_generation_pool else 0,
            'total_apples_deposited': sum([c.apples_deposited for c in creature_generation_pool]),
            'total_creatures_caught': sum([e.creatures_caught for e in enemy_generation_pool])
        }
        
        log_generation_stats(results_csv_path, stats)
        
        # Create next generation using the full generation pools
        creatures = reproduction(
            creature_generation_pool, cfg, cfg['creature_pop_size'],
            cfg['creature_hidden_size'], Creature
        )
        
        enemies = reproduction(
            enemy_generation_pool, cfg, cfg['enemy_pop_size'],
            cfg['enemy_hidden_size'], Enemy
        )

        print(f"Generation {generation}: {stats['creatures_survived']} creatures survived. Avg Fitness: {stats['avg_creature_fitness']:.2f}")

if __name__ == "__main__":
    run_headless_simulation(total_generations=100)