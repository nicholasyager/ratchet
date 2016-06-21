import logging
import socket
import struct

import io

import cv2
import numpy as np
import time


class Car:
    def __init__(self):
        self.telemetry = [0, 0, 0, 0, 1]
        self.telemetry_buffer = [0, 0, 0, 0, 1]

    def forward(self):
        self.telemetry[0] = 1
        self.telemetry[1] = 0

    def backward(self):
        self.telemetry[0] = 0
        self.telemetry[1] = 1

    def neutral(self):
        self.telemetry[0] = 0
        self.telemetry[1] = 0

    def left(self):
        self.telemetry[2] = 1
        self.telemetry[3] = 0

    def right(self):
        self.telemetry[2] = 0
        self.telemetry[3] = 1

    def straight(self):
        self.telemetry[2] = 0
        self.telemetry[3] = 0

    def model_active(self):
        self.telemetry[4] = 1

    def model_inactive(self):
        self.telemetry[4] = 0

    def has_update(self):
        update = False
        for index, value in enumerate(self.telemetry):
            if self.telemetry_buffer[index] != value:
                update = True
                self.telemetry_buffer[index] = value

        return update


class RemoteCar(Car):
    def __init__(self, ip_address):
        Car.__init__(self)
        self.ip_address = ip_address

        self.video_connection = None
        self.controls_connection = None

        self.connect_video()
        time.sleep(0.5)
        self.connect_controls()

    def connect_video(self):
        logging.info("Loading Video Feed.")
        video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        video_socket.connect((self.ip_address, 8000))
        self.video_connection = video_socket.makefile('rb')

    def connect_controls(self):
        logging.info("Initializing controls.")
        controls_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        controls_socket.connect((self.ip_address, 8001))
        self.controls_connection = controls_socket.makefile('wb')

    def write_controls(self, payload=None):
        if not payload:
            payload = bytes(' '.join(map(str, self.telemetry)),encoding="ascii")
        self.controls_connection.write(payload)
        self.controls_connection.flush()

    def get_frame(self):
        image_len = struct.unpack('<L', self.video_connection.read(struct.calcsize('<L')))[0]
        stream = io.BytesIO()
        stream.write(self.video_connection.read(image_len))
        stream.seek(0)  # Rewind stream
        return cv2.imdecode(np.fromstring(stream.getvalue(), dtype=np.uint8), 1)
