import pygame
import numpy as np
import sys
import time

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 4
CELL_SIZE = 80
MARGIN = 50

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (100, 149, 237)
GREEN = (152, 251, 152)
RED = (255, 99, 71)
DARK_BLUE = (65, 105, 225)
GOLD = (255, 215, 0)
LIGHT_BLUE = (173, 216, 230)

# Create the window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("❄️ FrozenLake AI Visualization 🎮")
clock = pygame.time.Clock()

class FrozenLakeVisualizer:
    def __init__(self):
        self.grid = [
            ['S', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'H'],
            ['F', 'F', 'F', 'H'],
            ['H', 'F', 'F', 'G']
        ]
        self.current_state = 0
        # Initialize Q-table with small random values for better learning
        self.q_table = np.random.uniform(-0.1, 0.1, (16, 4))
        self.episode = 0
        self.successes = 0
        self.steps = 0
        self.total_rewards = 0
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.training = False
        self.last_action = None
        
    def draw_grid(self):
        # Draw background
        screen.fill(LIGHT_BLUE)
        
        # Draw title
        title = self.font.render("❄️ FrozenLake AI Agent 🎮", True, DARK_BLUE)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 20))
        
        # Draw grid background
        grid_bg_rect = pygame.Rect(MARGIN - 10, 90, GRID_SIZE * CELL_SIZE + 20, GRID_SIZE * CELL_SIZE + 20)
        pygame.draw.rect(screen, BLUE, grid_bg_rect)
        pygame.draw.rect(screen, BLACK, grid_bg_rect, 3)
        
        # Draw grid cells
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x = MARGIN + col * CELL_SIZE
                y = MARGIN + 100 + row * CELL_SIZE
                state = row * GRID_SIZE + col
                
                # Determine cell color
                if self.grid[row][col] == 'S':
                    color = GREEN
                elif self.grid[row][col] == 'H':
                    color = RED
                elif self.grid[row][col] == 'G':
                    color = GOLD
                else:
                    color = WHITE
                
                # Draw cell
                pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 2)
                
                # Draw cell labels
                if self.grid[row][col] == 'S':
                    text = self.small_font.render("START", True, BLACK)
                elif self.grid[row][col] == 'G':
                    text = self.small_font.render("GOAL", True, BLACK)
                elif self.grid[row][col] == 'H':
                    text = self.small_font.render("HOLE", True, WHITE)
                else:
                    text = self.small_font.render(f"{state}", True, BLACK)
                
                screen.blit(text, (x + CELL_SIZE//2 - text.get_width()//2, 
                                 y + CELL_SIZE//2 - text.get_height()//2))
                
                # Draw agent
                if state == self.current_state:
                    pygame.draw.circle(screen, DARK_BLUE, 
                                     (x + CELL_SIZE//2, y + CELL_SIZE//2), 
                                     CELL_SIZE//3)
                    agent_text = self.small_font.render("AI", True, WHITE)
                    screen.blit(agent_text, (x + CELL_SIZE//2 - agent_text.get_width()//2,
                                           y + CELL_SIZE//2 - agent_text.get_height()//2))
                    
                    # Draw action indicator
                    if self.last_action is not None:
                        action_symbols = ["←", "↓", "→", "↑"]
                        action_text = self.small_font.render(action_symbols[self.last_action], True, RED)
                        screen.blit(action_text, (x + CELL_SIZE - 20, y + 5))
    
    def draw_info_panel(self):
        # Draw info panel background
        info_rect = pygame.Rect(WIDTH - 300, 100, 280, 400)
        pygame.draw.rect(screen, WHITE, info_rect)
        pygame.draw.rect(screen, BLACK, info_rect, 2)
        
        # Draw info text
        info_title = self.font.render("Agent Info", True, BLACK)
        screen.blit(info_title, (WIDTH - 290, 110))
        
        success_rate = (self.successes / max(1, self.episode)) * 100
        info_lines = [
            f"Episode: {self.episode}",
            f"Current State: {self.current_state}",
            f"Steps: {self.steps}",
            f"Total Rewards: {self.total_rewards:.2f}",
            f"Success Rate: {success_rate:.1f}%",
            f"Successes: {self.successes}/{self.episode}"
        ]
        
        for i, line in enumerate(info_lines):
            text = self.small_font.render(line, True, BLACK)
            screen.blit(text, (WIDTH - 280, 160 + i * 30))
        
        # Draw Q-values for current state
        q_title = self.small_font.render("Current Q-values:", True, BLACK)
        screen.blit(q_title, (WIDTH - 280, 340))
        
        actions = ["LEFT", "DOWN", "RIGHT", "UP"]
        for i, (action, q_value) in enumerate(zip(actions, self.q_table[self.current_state])):
            color = RED if i == self.last_action else BLACK
            q_text = self.small_font.render(f"{action}: {q_value:.3f}", True, color)
            screen.blit(q_text, (WIDTH - 280, 370 + i * 25))
    
    def draw_controls(self):
        controls = [
            "Controls:",
            "SPACE - Run one episode",
            "S - Take one step", 
            "R - Reset game",
            "T - Train (100 episodes)",
            "Q - Quit"
        ]
        
        for i, control in enumerate(controls):
            text = self.small_font.render(control, True, DARK_BLUE)
            screen.blit(text, (50, 400 + i * 25))
    
    def get_valid_actions(self, state):
        """Get valid actions from current state (considering boundaries)"""
        row, col = state // 4, state % 4
        valid_actions = []
        
        if col > 0:  # Can move left
            valid_actions.append(0)
        if row < 3:  # Can move down
            valid_actions.append(1)
        if col < 3:  # Can move right
            valid_actions.append(2)
        if row > 0:  # Can move up
            valid_actions.append(3)
            
        return valid_actions
    
    def step(self):
        state = self.current_state
        
        # Get valid actions
        valid_actions = self.get_valid_actions(state)
        if not valid_actions:
            return True  # No valid moves, episode ends
            
        # Choose action (epsilon-greedy)
        if np.random.random() < 0.2:  # Exploration
            action = np.random.choice(valid_actions)
        else:  # Exploitation
            # Only consider valid actions
            valid_q_values = [self.q_table[state][a] for a in valid_actions]
            best_action_idx = np.argmax(valid_q_values)
            action = valid_actions[best_action_idx]
        
        self.last_action = action
        
        # Calculate next state
        row, col = state // 4, state % 4
        if action == 0:  # Left
            col -= 1
        elif action == 1:  # Down
            row += 1
        elif action == 2:  # Right
            col += 1
        elif action == 3:  # Up
            row -= 1
        
        next_state = row * 4 + col
        cell_type = self.grid[row][col]
        
        # Calculate reward
        if cell_type == 'H':
            reward = -1  # Negative reward for falling in hole
            done = True
        elif cell_type == 'G':
            reward = 10  # Large positive reward for reaching goal
            done = True
            self.successes += 1
        else:
            reward = -0.01  # Small negative reward to encourage efficiency
            done = False
        
        # Update Q-table
        learning_rate = 0.1
        discount_factor = 0.95
        
        # Only consider valid next actions for max Q-value
        next_valid_actions = self.get_valid_actions(next_state)
        if next_valid_actions:
            max_next_q = max([self.q_table[next_state][a] for a in next_valid_actions])
        else:
            max_next_q = 0
            
        self.q_table[state][action] = self.q_table[state][action] + learning_rate * (
            reward + discount_factor * max_next_q - self.q_table[state][action]
        )
        
        # Update state
        self.current_state = next_state
        self.total_rewards += reward
        self.steps += 1
        
        return done
    
    def reset(self):
        self.current_state = 0
        self.steps = 0
        self.last_action = None
    
    def train(self, episodes=100):
        self.training = True
        print(f"Training for {episodes} episodes...")
        
        training_font = pygame.font.Font(None, 28)
        
        for episode in range(episodes):
            self.reset()
            done = False
            
            while not done:
                done = self.step()
            
            self.episode += 1
            
            # Update display every 10 episodes
            if episode % 10 == 0:
                self.draw()
                
                # Draw training progress
                progress = (episode + 1) / episodes
                progress_rect = pygame.Rect(50, 350, 300, 20)
                pygame.draw.rect(screen, WHITE, progress_rect)
                pygame.draw.rect(screen, BLACK, progress_rect, 2)
                
                fill_width = 300 * progress
                progress_fill = pygame.Rect(50, 350, fill_width, 20)
                pygame.draw.rect(screen, GREEN, progress_fill)
                
                progress_text = training_font.render(f"Training: {episode + 1}/{episodes}", True, BLACK)
                screen.blit(progress_text, (50, 330))
                
                pygame.display.flip()
                
                # Check for quit events during training
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        self.training = False
                        return
        
        self.training = False
        print("Training completed!")
    
    def draw(self):
        self.draw_grid()
        self.draw_info_panel()
        self.draw_controls()
    
    def run(self):
        running = True
        auto_play = False
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not self.training:
                        # Run one episode
                        self.reset()
                        done = False
                        while not done and not self.training:
                            done = self.step()
                            self.draw()
                            pygame.display.flip()
                            pygame.time.delay(300)  # Slow down for visualization
                            
                            # Check for interrupts
                            for e in pygame.event.get():
                                if e.type == pygame.QUIT:
                                    running = False
                                    break
                                elif e.type == pygame.KEYDOWN and e.key == pygame.K_q:
                                    running = False
                                    break
                            if not running:
                                break
                                
                        if running:
                            self.episode += 1
                    
                    elif event.key == pygame.K_s and not self.training:
                        # Take one step if not in terminal state
                        current_row, current_col = self.current_state // 4, self.current_state % 4
                        current_cell = self.grid[current_row][current_col]
                        if current_cell not in ['H', 'G']:
                            self.step()
                    
                    elif event.key == pygame.K_r and not self.training:
                        # Reset
                        self.reset()
                    
                    elif event.key == pygame.K_t and not self.training:
                        # Train in background
                        self.train(100)
                    
                    elif event.key == pygame.K_q:
                        running = False
            
            self.draw()
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        sys.exit()

# Run the visualizer
if __name__ == "__main__":
    visualizer = FrozenLakeVisualizer()
    visualizer.run()