"""Remote

Usage:
    remote.py [--dir DIR --from FROM --to TO --run NUM]

Options:
    --dir DIR       The directory to write the training data to.
    --from FROM     The starting point for the data.
    --to TO         The destination for the data.
    --run NUM       The run number for the training set.
"""

import logging
import os

from docopt import docopt
import cv2
import numpy as np
import pygame
from car import RemoteCar


def main(args):
    logging.basicConfig(level=logging.INFO)
    logging.info("Initialzing.")
    pygame.init()
    car = RemoteCar('192.168.0.110')

    logging.info("Reading first frame.")
    frame = car.get_frame()
    car.write_controls(payload=bytes('0 0 0 0 0', encoding='ascii'))

    logging.info("Configuring display.")
    screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
    pygame.display.set_caption('Ghost Car')
    pygame.mouse.set_visible(0)

    path = None
    file_num = 0
    if args['--from']:
        path = args['--dir']+'/'+args['--from'].replace(' ','_')+"."+args['--to'].replace(' ','_')+'.'+args['--run']
        if not os.path.exists(path):
            os.mkdir(path)

    while True:
        frame = car.get_frame()
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pygame.surfarray.blit_array(screen, np.rot90(np.fliplr(np.flipud(frameRGB)))[::-1])
        pygame.display.update()

        pygame.event.get()
        press = pygame.key.get_pressed()
        if press[pygame.K_UP] == 1:
            car.forward()
        elif press[pygame.K_DOWN] == 1:
            car.backward()
        else:
            car.neutral()

        if press[pygame.K_RIGHT] == 1:
            car.right()
        elif press[pygame.K_LEFT] == 1:
            car.left()
        else:
            car.straight()

        if press[pygame.K_q] == 1:
            car.model_inactive()

        if car.has_update():
            car.write_controls()

        if path:
            image_array = np.reshape(frame, (1, frame.shape[0] * frame.shape[1] * frame.shape[2]))
            telemetry_array = np.array(car.telemetry)
            np.savez(path+"/frame{0:05d}.npz".format(file_num), image_array, telemetry_array)
            file_num += 1

            if car.telemetry[4] == 0:
                logging.info("Ending session.")
                for _ in range(5):
                    image_array = np.reshape(frame, (1, frame.shape[0] * frame.shape[1] * frame.shape[2]))
                    telemetry_array = np.array(car.telemetry)
                    np.savez(path+"/frame{0:05d}.npz".format(file_num), image_array, telemetry_array)
                    file_num += 1
                return

if __name__ == '__main__':
    main(docopt(__doc__, version='Remote v0.0.1'))

