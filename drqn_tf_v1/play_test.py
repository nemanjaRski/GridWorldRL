from gridworld import GameEnv
import pygame
from PIL import Image

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)


def draw_text(text, surface, pos_x, pos_y, font_size=24, font_color=YELLOW):
    text_font = pygame.font.SysFont('arial', font_size)
    text_surf = text_font.render(text, True, font_color)
    surface.blit(text_surf, (pos_x, pos_y))

def choose_action():

    action = None
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                action = -1
            if event.key == pygame.K_UP:
                action = 0
            if event.key == pygame.K_DOWN:
                action = 1
            if event.key == pygame.K_LEFT:
                action = 2
            if event.key == pygame.K_RIGHT:
                action = 3
    return action



"""Game environment"""
env = GameEnv(partial=True, size=19, num_goals=20, num_fires=15, for_print=True)
state = env.reset()

display_width = 1024
display_height = 768

pygame.init()
screen = pygame.display.set_mode([display_width, display_height])


max_ep_length = 50
episode_reward = 0
num_of_red = 0
num_of_green = 0
num_of_stuck = 0
current_step = 0


while current_step < max_ep_length:

    # have to flip/rotate the image -- dunno why
    image = pygame.surfarray.make_surface(env.render_env())
    image = pygame.transform.rotate(image, -90) #rotate counter-clockwise
    image = pygame.transform.flip(image, True, False) #flip on Horizontal axis
    image = pygame.transform.scale(image, (display_width, display_height)) #rescale image to display size

    draw_text("Score: {:.2f}".format(episode_reward), image, 10, 10)
    screen.blit(image, (0, 0))

    action = choose_action()
    if action == -1:
        break

    reward = 0
    if action is not None:
        current_step += 1
        next_state, reward, done = env.step(action)

    pygame.display.update()
    episode_reward += reward


pygame.display.get_surface().fill(BLACK)
draw_text("Score: {:.2f}".format(episode_reward), pygame.display.get_surface(), 10, 10, font_size=128, font_color=WHITE)
pygame.display.update()

pygame.time.delay(3000)
pygame.quit()