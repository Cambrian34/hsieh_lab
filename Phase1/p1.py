import pygame
import numpy as np
import math
import random

# --- Constants & Colors ---
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
COLOR_BACKGROUND = (30, 30, 30)
COLOR_ORGANOID = (138, 43, 226)  # Purple
COLOR_TARGET = (50, 200, 50)     # Green
COLOR_TEXT = (255, 255, 255)

class LSTM_NeuralNetwork:
    """Placeholder LSTM Network (can be swapped for your SNN/Organoid API)"""
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size

        self.wf, self.bf = np.random.randn(concat_size, hidden_size) * 0.1, np.zeros((1, hidden_size))
        self.wi, self.bi = np.random.randn(concat_size, hidden_size) * 0.1, np.zeros((1, hidden_size))
        self.wc, self.bc = np.random.randn(concat_size, hidden_size) * 0.1, np.zeros((1, hidden_size))
        self.wo, self.bo = np.random.randn(concat_size, hidden_size) * 0.1, np.zeros((1, hidden_size))
        
        self.w_out, self.b_out = np.random.randn(hidden_size, output_size) * 0.1, np.zeros((1, output_size))
        
        self.reset_state()

    def reset_state(self):
        self.hidden_state = np.zeros((1, self.hidden_size))
        self.cell_state = np.zeros((1, self.hidden_size))

    def predict(self, inputs):
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

class StationaryOrganoid:
    def __init__(self, x, y, nn):
        self.pos = np.array([x, y], dtype=np.float64)
        self.angle = 0.0  # Facing right initially
        self.nn = nn
        self.fitness = 0.0
        self.max_turn_rate = 0.15
        self.radius = 20

    def get_inputs(self, target):
        """Returns the normalized relative angle to the target (-1.0 to 1.0)"""
        direction = target.pos - self.pos
        target_angle = math.atan2(direction[1], direction[0])
        relative_angle = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
        return np.array([relative_angle / math.pi])

    def update(self, target):
        # 1. Gather Sensory Input
        inputs = self.get_inputs(target)
        
        # 2. Get Neural Decision (Negative = Left, Positive = Right)
        # When bridging to your organoid, pass the `inputs` out via serial/API and await `turn_request`
        turn_request = self.nn.predict(inputs)[0] 
        
        # 3. Apply Rotation
        self.angle = (self.angle + turn_request * self.max_turn_rate) % (2 * math.pi)
        
        # 4. Calculate reward/fitness based on accuracy
        direction = target.pos - self.pos
        target_angle = math.atan2(direction[1], direction[0])
        relative_angle = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
        
        # Reward is highest (1.0) when facing exactly at the target
        facing_accuracy = 1.0 - (abs(relative_angle) / math.pi)
        self.fitness += facing_accuracy

    def draw(self, screen):
        # Draw the main body
        pygame.draw.circle(screen, COLOR_ORGANOID, self.pos.astype(int), self.radius)
        
        # Draw the visual "eye" / rotation indicator
        end_pos = self.pos + np.array([math.cos(self.angle), math.sin(self.angle)]) * (self.radius + 15)
        pygame.draw.line(screen, (255, 255, 255), self.pos.astype(int), end_pos.astype(int), 4)

class Target:
    def __init__(self, center_x, center_y, orbit_radius):
        # Spawn randomly in a circle around the center
        angle = random.uniform(0, 2 * math.pi)
        self.pos = np.array([center_x + math.cos(angle) * orbit_radius, 
                             center_y + math.sin(angle) * orbit_radius])
        self.radius = 12

    def draw(self, screen):
        pygame.draw.circle(screen, COLOR_TARGET, self.pos.astype(int), self.radius)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Organoid Rotation Training Environment")
    clock = pygame.time.Clock()


    # 1 Input (Relative Angle), 1 Output (Turn Left/Right)
    # The hidden size can be adjusted based on the complexity you want the placeholder to have
    nn = LSTM_NeuralNetwork(input_size=1, hidden_size=8, output_size=1)
    
    # Place agent dead center
    agent = StationaryOrganoid(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, nn)
    
    # Spawn initial target 150 pixels away
    target = Target(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, 150)
    
    target_timer = 0
    target_lifespan = 300  # Frames until target moves
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update logic
        agent.update(target)
        
        # Move the target occasionally to force the agent to keep tracking
        target_timer += 1
        if target_timer >= target_lifespan:
            target = Target(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, 150)
            target_timer = 0

        # Rendering
        screen.fill(COLOR_BACKGROUND)
        target.draw(screen)
        agent.draw(screen)
        
        
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main()