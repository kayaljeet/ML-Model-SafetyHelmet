# ML-Model-Tester

This project implements socket programming to establish a connection between a server and its clients. The clients send a video stream to the server, which retrieves a machine learning model from a specified path. The server then runs the model on the incoming video stream, displaying the inference frames in an imshow window with bounding boxes.

This repository includes several pre-trained model files in ".pt" format, which were trained using the [ML-Model-Trainer](https://github.com/kayaljeet/ML-Model-Maker.git) application.

## Contents

### 1. Client
- **a) `client.py`:**  
  Socket program to send a video stream to a server program.
- **b) `config.txt`:**  
  Configuration file for the client.
- **c) `helmet.mp4`:**  
  Sample video to test the "helmet.pt" model.

### 2. Server
- **a) `server.py`:**  
  Socket program to receive video streams from clients and implement object recognition using a PyTorch file of a pre-trained model.
