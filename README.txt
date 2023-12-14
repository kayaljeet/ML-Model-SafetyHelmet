ML-Model-Tester

This project implements socket programming to establish a connection between a Sever and its Clients, where, the clients send a video stream to the Server and the Server gets a ML model from a given path, and runs it on incoming video stream, displaying the inference frames on a imshow window with bounding boxes.
This Repository consists of a few pre trained model files in ".pt" format, that were trained using the ModelTrainer project application.

Contents:

1) Client
	a) client.py:
		Socket program to send a video stream to a Server program
	b) config.txt
	c) helmet.mp4:
		sample video to test "helmet.pt" model
2) Server 
	a) server.py:
		Socket program to receive video streams from Clients and implement object recognition, using a PyTorch file of a pre-trained model
