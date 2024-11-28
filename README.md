# 42_AlgoGenuises_Secuirty-Survelliance
Smart Survelliance System 

This project implements an AI-powered CCTV surveillance system using Deep Learning and Computer Vision technologies to monitor and regulate mobile phone usage in restricted areas such as parliaments, classrooms, meeting rooms, and other secure zones. Built with TensorFlow/Keras for model training and OpenCV for video frame processing, the system uses a pre-trained Convolutional Neural Network (CNN) to detect mobile phone usage in real-time.

When phone usage is identified, the system utilizes Twilio API (or any messaging service) to send an automated alert to the user as a warning. If the user continues to use their phone or exceeds a predefined threshold, the incident is logged and escalated to an administrator through notifications.

The project also includes integration of YOLO  or similar object detection frameworks for accurate recognition of individuals and their actions within CCTV footage. Designed with scalability in mind, it supports real-time video feed processing and leverages AI-based analytics for actionable insights. This solution ensures enhanced productivity, strict policy enforcement, and improved security in critical environments.
