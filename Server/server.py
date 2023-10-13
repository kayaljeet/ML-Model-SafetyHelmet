import argparse
import pickle
import struct
from _thread import *
import numpy as np
import cv2
import socket
import torch
import ssl
import time

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser(description="Your Description Here")
parser.add_argument("--path", required=True, help="Path to the YOLOv5 model")
args = parser.parse_args()
path = args.path

model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)

host = '0.0.0.0'
port = 8080


# Define a timeout value (in seconds)
CLIENT_TIMEOUT = 10

def client_handler(connection):
    data = b""
    payload_size = struct.calcsize("Q")
    last_data_time = time.time()  # Initialize the last data time

    while True:
        try:
            while len(data) < payload_size:
                packet = connection.recv(4 * 1024)
                if not packet:
                    break
                data += packet

            if not data:
                # No data received within the timeout period, close the connection
                current_time = time.time()
                if current_time - last_data_time > CLIENT_TIMEOUT:
                    print(f"Client {camera_name} stopped sending data. Closing the connection.")
                    break

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            while len(data) < msg_size:
                data += connection.recv(4 * 1024)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            camera_name, frame = pickle.loads(frame_data)
            results = model(frame)
            frame = np.squeeze(results.render())
            cv2.imshow(camera_name, frame)
            cv2.waitKey(1)

            # Update the last data time when data is received
            last_data_time = time.time()

        except Exception as e:
            print(f"Client {camera_name} disconnected - exception {e}")
            cv2.destroyWindow(camera_name)
            for i in range(4):
                cv2.waitKey(1)
            break

    print("before connection close")
    return 0


def accept_connections(ServerSocket):
   Client, address = ServerSocket.accept()
   print('Connected to: ' + address[0] + ':' + str(address[1]))
   start_new_thread(client_handler, (Client,))
   #client_handler(Client)


def start_server(host, port):
    ServerSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        ServerSocket.bind((host, port))
    except socket.error as e:
        print(str(e))
    print(f'Server is listening on port {port}...')
    ServerSocket.listen()

    while True:
        accept_connections(ServerSocket)


start_server(host, port)
