# Image Processing Navigation Robot

Welcome to the Team Spectro's Image Processing Robot repository! This project involves the development of a robot capable of autonomous navigation using image processing techniques. The robot utilizes an ESP32-CAM module for image acquisition, processing, and communication.


## Images

|![Exmo-bot (1)](https://github.com/sithija-vihanga/Image-processing-navigation-bot/assets/116638289/b0201a47-a1e2-488d-a0fe-2dfb9f20c80a) | ![Exmo-bot (3)](https://github.com/sithija-vihanga/Image-processing-navigation-bot/assets/116638289/d142f63f-d02e-40a1-b795-2aacd4ca4284)|
|:---------------------------------------------------:|:---------------------------------------------------:|
|![Exmo-bot (2)](https://github.com/sithija-vihanga/Image-processing-navigation-bot/assets/116638289/2771492d-50bf-477a-9f16-1d984ec340ec)| ![Exmo-bot (4)](https://github.com/sithija-vihanga/Image-processing-navigation-bot/assets/116638289/c5fbec38-7ab9-4aa2-82df-c9246cf936fd)|


## Features

- **Image Processing**: The robot employs Python code to process the camera feed, enabling it to identify objects, estimate distances, and detect shapes like arrows.

- **Wireless Communication**: The robot connects to a local area network, usually a hotspot, to establish a connection with a PC or laptop. The camera feed is transmitted to the computer via WiFi.

- **Automatic Navigation**: Using the results of image processing, the robot makes decisions for autonomous navigation. Commands are sent back to the robot to control its movement.

- **Bluetooth Navigation**: The robot is equipped with Bluetooth navigation, enhancing its maneuverability and ease of control.

- **Mechanum Wheels**: The use of mechanum wheels enhances the robot's mobility, allowing it to move smoothly in various directions.

- **Custom Chassis**: The robot's chassis is designed using SolidWorks and fabricated through laser cutting, ensuring a sturdy and functional design.

## Getting Started

To get started with the Spectro Image Processing Robot, follow these steps:

1. Clone this repository to your local machine.
2. Make sure you have Python installed, and install the necessary dependencies listed in `requirements.txt`.
3. Open the Python code for image processing and update the IP address to match the robot's IP address. This IP address will be displayed on the robot's OLED display.
4. Connect your PC or laptop to the robot's hotspot.
5. Run the Python code to establish a connection and start receiving the camera feed.
6. Observe the robot's navigation decisions and control it through the provided commands.

## Folder Structure

- `code/`: Contains the Python code for image processing and communication.
- `Chassis Design/`: Includes SolidWorks design files for the custom robot chassis.
- `Images/`: Stores images related to the project.
- `README.md`: You are here! This README file.

## Contributing

Contributions to this project are welcome! If you find a bug or have an enhancement in mind, please open an issue or submit a pull request.


