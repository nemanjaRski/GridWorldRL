import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import skimage.transform as transform


class gameOb():
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name


class gameEnv():
    def __init__(self, partial, size, num_goals=20, num_fires=10):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial

        self.num_goals = num_goals
        self.num_fires = num_fires

        self.reset()

    def reset(self):

        self.objects = []

        self.objects.append(gameOb(self.newPosition(), 1, 1, 2, None, 'hero'))

        for n in range(self.num_goals):
            self.objects.append(gameOb(self.newPosition(), 1, 1, 1, 1, 'goal'))

        for n in range(self.num_fires):
            self.objects.append(gameOb(self.newPosition(), 1, 1, 0, -1, 'fire'))

        self.state = self.renderEnv()
        return self.state

    def moveChar(self, direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        hero = self.objects[0]

        heroX = hero.x
        heroY = hero.y
        penalize = 0.

        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY - 2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX - 2:
            hero.x += 1
        if hero.x == heroX and hero.y == heroY:
            penalize = -1

        self.objects[0] = hero

        return penalize

    def newPosition(self):

        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        currentPositions = []

        for t in itertools.product(*iterables):
            points.append(t)

        for objectA in self.objects:
            if (objectA.x, objectA.y) not in currentPositions:
                currentPositions.append((objectA.x, objectA.y))

        for pos in currentPositions:
            points.remove(pos)

        location = np.random.choice(range(len(points)), replace=False)

        return points[location]

    def checkGoal(self):
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
                    self.objects.append(gameOb(self.newPosition(), 1, 1, 1, 1, 'goal'))
                    return other.reward, False
                else:
                    self.objects.append(gameOb(self.newPosition(), 1, 1, 0, -1, 'fire'))
                    return other.reward, False
        return -0.1, False

    def renderEnv(self, print=False):
        # a = np.zeros([self.sizeY,self.sizeX,3])
        a = np.ones([self.sizeY + 2, self.sizeX + 2, 3])
        a[1:-1, 1:-1, :] = 0
        hero = None
        for item in self.objects:
            a[item.y + 1:item.y + item.size + 1, item.x + 1:item.x + item.size + 1, item.channel] = item.intensity
            if item.name == 'hero':
                hero = item
        if self.partial == True:
            a = a[hero.y:hero.y + 3, hero.x:hero.x + 3, :]
        # b = scipy.misc.imresize(a[:,:,0],[84,84,1],interp='nearest') np.array(Image.fromarray(a[:,:,0]).resize())
        # c = scipy.misc.imresize(a[:,:,1],[84,84,1],interp='nearest')
        # d = scipy.misc.imresize(a[:,:,2],[84,84,1],interp='nearest')
        # b = np.array(Image.fromarray(a[:, :, 0]).resize(size=(84, 84)))
        # c = np.array(Image.fromarray(a[:, :, 1]).resize(size=(84, 84)))
        # d = np.array(Image.fromarray(a[:, :, 2]).resize(size=(84, 84)))
        # print(b.shape)
        # a = np.stack([b, c, d], axis=2)

        a = (transform.resize(a, [84, 84, 3], order=0, preserve_range=True) * 255).astype(np.uint8)

        return a

    def step(self, action):
        penalty = self.moveChar(action)
        reward, done = self.checkGoal()
        state = self.renderEnv()

        if reward is None:
            print(done)
            print(reward)
            print(penalty)
            return state, (reward + penalty), done
        else:
            return state, (reward + penalty), done
