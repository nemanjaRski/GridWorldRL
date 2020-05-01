import numpy as np
import itertools
import skimage.transform as transform


class GameOb:
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name


class GameEnv:
    def __init__(self, partial, size, num_goals=20, num_fires=10, for_print=False, sight=1):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial

        self.sight = int(sight)
        if self.sight * 2 > self.sizeX:
            self.sight = int(self.sizeX / 2)

        self.num_goals = num_goals
        self.num_fires = num_fires
        self.for_print = for_print

        self.reset()

    def reset(self):

        self.objects = []

        self.objects.append(GameOb(self.new_position(), 1, 1, 2, None, 'hero'))

        for n in range(self.num_goals):
            self.objects.append(GameOb(self.new_position(), 1, 1, 1, 1, 'goal'))

        for n in range(self.num_fires):
            self.objects.append(GameOb(self.new_position(), 1, 1, 0, -1, 'fire'))

        self.state = self.render_env()
        return self.state

    def move_char(self, direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        hero = self.objects[0]

        hero_x = hero.x
        hero_y = hero.y
        penalize = 0.

        if direction == 0 and hero.y >= self.sight:
            hero.y -= 1
        if direction == 1 and hero.y < self.sizeY - 2 + self.sight:
            hero.y += 1
        if direction == 2 and hero.x >= self.sight:
            hero.x -= 1
        if direction == 3 and hero.x < self.sizeX - 2 + self.sight:
            hero.x += 1
        if hero.x == hero_x and hero.y == hero_y:
            penalize = 0

        self.objects[0] = hero

        return penalize

    def new_position(self):

        iterables = [range(self.sight - 1, self.sizeX + self.sight), range(self.sight - 1, self.sizeY + self.sight)]
        points = []
        current_positions = []

        for t in itertools.product(*iterables):
            points.append(t)

        for objectA in self.objects:
            if (objectA.x, objectA.y) not in current_positions:
                current_positions.append((objectA.x, objectA.y))

        for pos in current_positions:
            points.remove(pos)

        location = np.random.choice(range(len(points)), replace=False)
        return points[location]

    def check_goal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        ended = False
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(GameOb(self.new_position(), 1, 1, 1, 1, 'goal'))
                    return other.reward, False
                else:
                    self.objects.append(GameOb(self.new_position(), 1, 1, 0, -1, 'fire'))
                    return other.reward, False
        return -0.1, False

    def render_env(self):
        a = np.ones([self.sizeY + 2 * self.sight, self.sizeX + 2 * self.sight, 3])
        a[self.sight:-self.sight, self.sight:-self.sight, :] = 0

        hero = None
        for item in self.objects:
            a[item.y + 1:item.y + item.size + 1, item.x + 1:item.x + item.size + 1, item.channel] = item.intensity
            if item.name == 'hero':
                hero = item

        if self.partial:
            a = a[hero.y + 1 - self.sight:hero.y + 2 + self.sight, hero.x + 1 - self.sight:hero.x + 2 + self.sight, :]

        if self.for_print:
            a = (transform.resize(a, [84, 84, 3], order=0, preserve_range=True) * 255).astype(np.uint8)
        else:
            a = (transform.resize(a, [84, 84, 3], order=0, preserve_range=True)).astype(np.uint8)

        return a

    def render_full_env(self):
        a = np.ones([self.sizeY + 2, self.sizeX + 2, 3])
        a[1:-1, 1:-1, :] = 0
        for item in self.objects:
            a[item.y + 1:item.y + item.size + 1, item.x + 1:item.x + item.size + 1, item.channel] = item.intensity
        a = (transform.resize(a, [84, 84, 3], order=0, preserve_range=True) * 255).astype(np.uint8)
        return a

    def step(self, action):
        penalty = self.move_char(action)
        reward, done = self.check_goal()
        state = self.render_env()

        if reward is None:
            print(done)
            print(reward)
            print(penalty)
            return state, (reward + penalty), done
        else:
            return state, (reward + penalty), done
