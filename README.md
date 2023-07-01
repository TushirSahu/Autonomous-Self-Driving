# Autonomous-Self-Driving 🚗
<img src="https://github.com/TushirSahu/Autonomous-Self-Driving/assets/96677478/3c0d3d3d-61ff-40c1-978a-0183b2e76c1b" alt="Video Preview" width="800" height="550">

## Description
This project aims to develop a self-driving car using deep learning techniques. The model is trained to predict steering angles based on input images from the car's cameras, allowing it to navigate autonomously.

## Dataset
The training data for this project was collected using a combination of real-world driving and simulation. The dataset consists of images captured from multiple cameras mounted on the car, along with corresponding steering angles. Unfortunately, the dataset used in this project is not publicly available.

## Model Architecture
The self-driving model utilizes the NVIDIA End-to-End model architecture. It consists of convolutional layers for feature extraction, followed by fully connected layers for regression. The model is trained to directly predict the steering angle given input images.

## Training
The model was trained using a combination of the Adam optimizer and mean squared error (MSE) loss function. The training data was augmented using techniques such as random image flips, zooms, and brightness adjustments. The model was trained for 20 epochs on a GPU, with a batch size of 100.The trained model achieved an MSE of 0.05 on the validation set, indicating accurate steering angle predictions. The model was able to navigate through various driving scenarios, including straight roads, curves, and intersections, demonstrating its effectiveness in autonomous driving tasks.

## Usage
1. Prepare the dataset:
- Collect driving data including images and corresponding steering angles.
- Split the dataset into training and validation sets.

2. Preprocess the data:
- Resize the images to the desired input size.
- Normalize the pixel values.
- Augment the dataset if necessary (e.g., random flips, zooms, brightness adjustments).

3. Train the model:
- Configure the model architecture and hyperparameters.
- Run the training script:
  ```
  python train.py --data-dir /path/to/dataset --epochs 20
  ```

4. Evaluate the model:
- Assess the performance of the trained model on the validation set.
- Calculate evaluation metrics such as mean squared error (MSE) or accuracy.

5. Test the model:
- Use the trained model to make predictions on unseen test data.
- Evaluate the model's performance on real-world scenarios.

6. Autonomous driving:
## Udacity Self-Driving Car Simulator

The Udacity Self-Driving Car Simulator is a tool provided by Udacity for testing and evaluating autonomous driving algorithms. It provides a virtual environment where you can simulate driving scenarios and collect data for training and testing self-driving models.

### Usage Instructions

1. **Download the Simulator:** You can download the Udacity Self-Driving Car Simulator from the official repository [here](https://github.com/udacity/self-driving-car-sim).

2. **Select a Mode:** The simulator offers two modes: Training and Autonomous. In the Training mode, you can manually drive the car in different scenarios and collect training data, including images and steering angles. In the Autonomous mode, you can test your trained self-driving models by providing them with the collected data.

3. **Configure the Simulator:** Before running the simulator, make sure to adjust the settings according to your requirements. You can modify parameters such as screen resolution, graphics quality, and data collection frequency.

4. **Start the Simulator:** Launch the Udacity Self-Driving Car Simulator and select the desired mode. In the Training mode, you can drive the car using keyboard inputs or a steering wheel controller. In the Autonomous mode, you can load your trained model and let it drive the car based on the collected data.

5. **Analyze Results:** After running the simulator, you can analyze the performance of your self-driving model. The simulator provides visualizations of the car's trajectory, sensor data, and predicted steering angles. You can use this information to evaluate the model's behavior and make improvements if necessary.

- Run the autonomous driving script:
  ```
  python drive.py
  ```


## Future Work
There are several areas for future improvement in this project:
- Fine-tuning the model architecture and hyperparameters for better performance.
- Collecting a larger and more diverse dataset to improve the model's generalization.
- Implementing advanced perception and control algorithms for handling complex driving scenarios.
- Integrating real-time object detection and tracking for enhanced safety and situational awareness.

