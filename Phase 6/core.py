import numpy as np
import random
import math
import os
from datetime import datetime
from numba import njit

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


def run_experiment_job(job_package):
    """
    A simple wrapper function that unpacks a config and runs a simulation.
    This is the function that multiprocessing.Pool.map will call.
    """
    try:
        name = job_package['name']
        config = job_package['config']
        generations = job_package['generations']
        
        # Create a unique results path
        results_path = f"results/results_{name}.csv"
        
        # Ensure the 'results' directory exists
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        print(f"[Process {os.getpid()}] STARTING job: {name}")
        
        # Call the main simulation function
        run_headless_simulation(
            config=config,
            total_generations=generations,
            results_csv_path=results_path
        )
        
        print(f"[Process {os.getpid()}] FINISHED job: {name}")
        return f"Job {name} complete."
    except Exception as e:
        print(f"[Job {name} FAILED] Error: {e}")
        import traceback
        traceback.print_exc()
        return f"Job {name} FAILED."


# --- JIT-Compiled Helper Functions ---

@njit
def _fast_move(pos, velocity, angle, speed, turn_request, accel_request):
    angle += turn_request * 0.1
    angle %= (2 * math.pi)
    speed += accel_request * 0.1
    speed = max(0.0, min(speed, 3.0))
    direction_x = math.cos(angle)
    direction_y = math.sin(angle)
    velocity[0] = direction_x * speed
    velocity[1] = direction_y * speed
    new_pos = pos + velocity
    new_pos[0] = max(WORLD_PADDING, min(new_pos[0], SCREEN_WIDTH - WORLD_PADDING))
    new_pos[1] = max(WORLD_PADDING, min(new_pos[1], SCREEN_HEIGHT - WORLD_PADDING))
    return new_pos, velocity, angle, speed

@njit
def _fast_find_nearest(my_pos, target_positions):
    if target_positions.shape[0] == 0:
        return -1, np.inf
    min_dist_sq = np.inf
    nearest_index = -1
    for i in range(target_positions.shape[0]):
        dx = my_pos[0] - target_positions[i, 0]
        dy = my_pos[1] - target_positions[i, 1]
        dist_sq = dx*dx + dy*dy
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            nearest_index = i
    return nearest_index, np.sqrt(min_dist_sq)

@njit
def _fast_enemy_attack(enemy_pos, creature_positions, creature_hiding_status, attack_range_sq):
    if creature_positions.shape[0] == 0:
        return -1
    for i in range(creature_positions.shape[0]):
        if not creature_hiding_status[i]:
            dx = enemy_pos[0] - creature_positions[i, 0]
            dy = enemy_pos[1] - creature_positions[i, 1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < attack_range_sq:
                return i
    return -1

# --- Simulation Classes ---

class LSTM_NeuralNetwork:
    # (Content is identical to the previous Numba-optimized version)
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.input_size = input_size # Store input size
        self.output_size = output_size # Store output size

        self.Wii = np.random.randn(hidden_size, input_size) * 0.01
        self.Whi = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bi = np.zeros((hidden_size, 1))
        self.Wif = np.random.randn(hidden_size, input_size) * 0.01
        self.Whf = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))
        self.Wig = np.random.randn(hidden_size, input_size) * 0.01
        self.Whg = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bg = np.zeros((hidden_size, 1))
        self.Wio = np.random.randn(hidden_size, input_size) * 0.01
        self.Who = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bo = np.zeros((hidden_size, 1))
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.by = np.zeros((output_size, 1))
        self.reset_state()

    def reset_state(self):
        self.h = np.zeros((self.hidden_size, 1))
        self.c = np.zeros((self.hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def predict(self, inputs):
        if len(inputs) != self.input_size:
            # This check will catch the error definitively
            raise ValueError(f"Network expected {self.input_size} inputs, but got {len(inputs)}")
            
        x = np.array(inputs).reshape(-1, 1)
        i = self.sigmoid(np.dot(self.Wii, x) + np.dot(self.Whi, self.h) + self.bi)
        f = self.sigmoid(np.dot(self.Wif, x) + np.dot(self.Whf, self.h) + self.bf)
        g = np.tanh(np.dot(self.Wig, x) + np.dot(self.Whg, self.h) + self.bg)
        o = self.sigmoid(np.dot(self.Wio, x) + np.dot(self.Who, self.h) + self.bo)
        self.c = f * self.c + i * g
        self.h = o * np.tanh(self.c)
        y = np.dot(self.Why, self.h) + self.by
        return np.tanh(y).flatten()

    def get_weights(self):
        return np.concatenate([
            self.Wii.flatten(), self.Whi.flatten(), self.bi.flatten(),
            self.Wif.flatten(), self.Whf.flatten(), self.bf.flatten(),
            self.Wig.flatten(), self.Whg.flatten(), self.bg.flatten(),
            self.Wio.flatten(), self.Who.flatten(), self.bo.flatten(),
            self.Why.flatten(), self.by.flatten()
        ])

    def set_weights(self, weights):
        """
        *** BUG FIX ***
        This method was not correctly re-assigning the new weights.
        This version now correctly unpacks the new weight list.
        """
        idx = 0
        def get_slice(shape):
            nonlocal idx
            size = np.prod(shape)
            result = weights[idx:idx + size]
            idx += size
            return result.reshape(shape)

        # List of all weight matrices in their correct order
        matrices = [
            self.Wii, self.Whi, self.bi,
            self.Wif, self.Whf, self.bf,
            self.Wig, self.Whg, self.bg,
            self.Wio, self.Who, self.bo,
            self.Why, self.by
        ]
        
        new_matrices = []
        try:
            for m in matrices:
                new_matrices.append(get_slice(m.shape))
        except ValueError as e:
            print(f"Error in set_weights: {e}. Check crossover/mutation logic.")
            return

        # Unpack the new matrices back into the instance variables
        (
            self.Wii, self.Whi, self.bi,
            self.Wif, self.Whf, self.bf,
            self.Wig, self.Whg, self.bg,
            self.Wio, self.Who, self.bo,
            self.Why, self.by
        ) = new_matrices


    def mutate(self, rate, strength):
        weights = self.get_weights()
        mask = np.random.random(weights.shape) < rate
        mutation = np.random.randn(mask.sum()) * strength
        weights[mask] += mutation
        self.set_weights(weights)

class BaseAgent:
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
        object_positions = np.array([obj.pos for obj in objects])
        nearest_index, min_dist = _fast_find_nearest(self.pos, object_positions)
        if nearest_index == -1:
            return None, float('inf')
        return objects[nearest_index], min_dist

    def move(self, turn_request, accel_request):
        new_pos, new_velocity, new_angle, new_speed = _fast_move(
            self.pos, self.velocity, self.angle, self.speed, turn_request, accel_request
        )
        self.pos = new_pos
        self.velocity = new_velocity
        self.angle = new_angle
        self.speed = new_speed

class Creature(BaseAgent):
    """
    *** CONFIRM THIS SECTION ***
    NUM_INPUTS must be 12.
    get_inputs must return 12 items.
    """
    NUM_INPUTS = 12 
    NUM_OUTPUTS = 5
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
        inputs = self.get_inputs(apples, heavy_apples, hiding_spots, score_zone, enemies, all_creatures)
        outputs = self.nn.predict(inputs)
        turn, accel, gather, self.communication_signal, hide_request = outputs
        
        if gather > 0 and self.apples_held < self.max_apples:
            all_apples = apples + heavy_apples
            nearest_apple, dist = self.get_nearest_object(all_apples)
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
                    self.fitness += 1
                    self.energy = min(self.energy, self.config['max_energy'])

        dist_to_score = np.linalg.norm(self.pos - score_zone.pos)
        if self.apples_held > 0 and dist_to_score < SCORE_ZONE_SIZE:
            self.fitness += self.apples_held * 10
            self.apples_deposited += self.apples_held
            self.apples_held = 0

        if hide_request > 0:
            nearest_spot, dist = self.get_nearest_object(hiding_spots)
            self.is_hiding = nearest_spot and dist < HIDING_SPOT_SIZE
        else:
            self.is_hiding = False
        
        self.move(turn, accel)
        self.energy -= self.config['energy_decay_rate']
        self.lifetime += 1
        return True

    def get_inputs(self, apples, heavy_apples, hiding_spots, score_zone, enemies, all_creatures):
        """
        *** CONFIRM THIS METHOD ***
        This implementation correctly returns 12 inputs.
        2 (apple) + 2 (zone) + 2 (enemy) + 2 (spot) + 4 (internal) = 12
        """
        def get_relative_info(target_object):
            if target_object:
                dist = np.linalg.norm(target_object.pos - self.pos)
                angle = math.atan2(target_object.pos[1] - self.pos[1], target_object.pos[0] - self.pos[0])
                return [dist / SCREEN_WIDTH, angle / math.pi]
            return [1.0, 0.0]
        
        nearest_apple, _ = self.get_nearest_object(apples + heavy_apples)
        inputs = get_relative_info(nearest_apple) # 2
        
        inputs.extend(get_relative_info(score_zone)) # 4
        
        nearest_enemy, _ = self.get_nearest_object(enemies)
        inputs.extend(get_relative_info(nearest_enemy)) # 6
        
        nearest_spot, _ = self.get_nearest_object(hiding_spots)
        inputs.extend(get_relative_info(nearest_spot)) # 8
        
        inputs.extend([
            self.energy / self.config['max_energy'],
            self.apples_held / self.max_apples,
            self.speed / 3.0,
            float(self.is_hiding)
        ]) # 12
        
        return inputs

class Enemy(BaseAgent):
    NUM_INPUTS = 8 
    NUM_OUTPUTS = 3
    def __init__(self, x, y, nn, config):
        super().__init__(x, y, nn, config)
        self.attack_cooldown = 0
        self.creatures_caught = 0
        self.attack_range_sq = (ENEMY_SIZE + CREATURE_SIZE) ** 2

    def update(self, creatures, hiding_spots, all_enemies):
        inputs = self.get_inputs(creatures, hiding_spots, all_enemies)
        outputs = self.nn.predict(inputs)
        self.move(outputs[0], outputs[1])
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        
        if outputs[2] > 0 and self.attack_cooldown == 0 and creatures:
            creature_positions = np.array([c.pos for c in creatures])
            creature_hiding_status = np.array([c.is_hiding for c in creatures])
            target_index = _fast_enemy_attack(
                self.pos, creature_positions, creature_hiding_status, self.attack_range_sq
            )
            if target_index != -1:
                target_creature = creatures[target_index]
                target_creature.energy -= self.config['enemy_attack_damage']
                if target_creature.energy <= 0:
                    self.fitness += self.config['enemy_fitness_bonus']
                    self.creatures_caught += 1
                self.attack_cooldown = 20
        return True

    def get_inputs(self, creatures, hiding_spots, all_enemies):
        # This method returns 8 inputs, matching NUM_INPUTS = 8
        def get_relative_info(target_object):
            if target_object:
                dist = np.linalg.norm(target_object.pos - self.pos)
                angle = math.atan2(target_object.pos[1] - self.pos[1], target_object.pos[0] - self.pos[0])
                return [dist / SCREEN_WIDTH, angle / math.pi]
            return [1.0, 0.0]
        visible_creatures = [c for c in creatures if not c.is_hiding]
        nearest_creature, _ = self.get_nearest_object(visible_creatures)
        inputs = get_relative_info(nearest_creature) # 2
        nearest_spot, _ = self.get_nearest_object(hiding_spots)
        inputs.extend(get_relative_info(nearest_spot)) # 4
        inputs.extend([
            self.speed / 3.0,
            self.attack_cooldown / 20.0,
            len(visible_creatures) / (len(creatures) + 1e-6),
            self.creatures_caught / 10.0
        ]) # 8
        return inputs

class Apple:
    def __init__(self, x, y): self.pos = np.array([x, y])
class HeavyApple:
    def __init__(self, x, y): self.pos = np.array([x, y])
class ScoreZone:
    def __init__(self, x, y): self.pos = np.array([x, y])
class HidingSpot:
    def __init__(self, x, y): self.pos = np.array([x, y])

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
    cfg = {**DEFAULT_CONFIG, **config}
    setup_csv_logger(results_csv_path)
    
    score_zone = ScoreZone(SCREEN_WIDTH/2, SCREEN_HEIGHT/2)
    hiding_spots = [
        HidingSpot(SCREEN_WIDTH/4, SCREEN_HEIGHT/4),
        HidingSpot(3*SCREEN_WIDTH/4, SCREEN_HEIGHT/4),
        HidingSpot(SCREEN_WIDTH/4, 3*SCREEN_HEIGHT/4),
        HidingSpot(3*SCREEN_WIDTH/4, 3*SCREEN_HEIGHT/4)
    ]
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
    
    _fast_move(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.0, 0.0, 0.1, 0.1)
    _fast_find_nearest(np.array([0.0, 0.0]), np.array([[1.0, 1.0], [2.0, 2.0]]))
    _fast_enemy_attack(np.array([0.0, 0.0]), np.array([[1.0, 1.0]]), np.array([False]), 100.0)

    for generation in range(1, total_generations + 1):
        creature_generation_pool = list(creatures)
        enemy_generation_pool = list(enemies)
        apples = [Apple(
            random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
            random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING)
        ) for _ in range(cfg['max_apples'])]
        heavy_apples = [HeavyApple(
            random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
            random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING)
        ) for _ in range(cfg['max_heavy_apples'])]

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
                else:
                    agent.creatures_caught = 0
                agent.nn.reset_state()

        for _ in range(cfg['generation_time']):
            for i in range(len(creatures) - 1, -1, -1):
                c = creatures[i]
                if not c.update(apples, heavy_apples, hiding_spots, score_zone, enemies, creatures):
                    creatures.pop(i) 
            if not creatures:
                break
            for e in enemies:
                e.update(creatures, hiding_spots, enemies)

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
        
        creatures = reproduction(
            creature_generation_pool, cfg, cfg['creature_pop_size'],
            cfg['creature_hidden_size'], Creature
        )
        enemies = reproduction(
            enemy_generation_pool, cfg, cfg['enemy_pop_size'],
            cfg['enemy_hidden_size'], Enemy
        )

        if generation % 10 == 0 or generation == 1:
             print(f"[{results_csv_path}] Gen {generation}: {stats['creatures_survived']} survived. Avg Fitness: {stats['avg_creature_fitness']:.2f}")

if __name__ == "__main__":
    print("Running a single test simulation...")
    run_headless_simulation(total_generations=100, results_csv_path="results/single_test_run.csv")

