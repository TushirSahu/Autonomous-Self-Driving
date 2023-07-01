# Autonomous-Self-Driving
<img src="https://github.com/TushirSahu/Autonomous-Self-Driving/assets/96677478/3c0d3d3d-61ff-40c1-978a-0183b2e76c1b" alt="Video Preview" width="800" height="550">

## Description
This project aims to develop a self-driving car using deep learning techniques. The model is trained to predict steering angles based on input images from the car's cameras, allowing it to navigate autonomously.

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
- Deploy the trained model on the self-driving car platform.
- Ensure all necessary hardware components (e.g., cameras, sensors) are properly connected.
- Run the autonomous driving script:
  ```
  python drive.py
  ```

## Dataset
The training data for this project was collected using a combination of real-world driving and simulation. The dataset consists of images captured from multiple cameras mounted on the car, along with corresponding steering angles. Unfortunately, the dataset used in this project is not publicly available.

## Model Architecture
The self-driving model utilizes the NVIDIA End-to-End model architecture. It consists of convolutional layers for feature extraction, followed by fully connected layers for regression. The model is trained to directly predict the steering angle given input images.

## Training
The model was trained using a combination of the Adam optimizer and mean squared error (MSE) loss function. The training data was augmented using techniques such as random image flips, zooms, and brightness adjustments. The model was trained for 20 epochs on a GPU, with a batch size of 100.

## Results
The trained model achieved an MSE of 0.05 on the validation set, indicating accurate steering angle predictions. The model was able to navigate through various driving scenarios, including straight roads, curves, and intersections, demonstrating its effectiveness in autonomous driving tasks.

## Future Work
There are several areas for future improvement in this project:
- Fine-tuning the model architecture and hyperparameters for better performance.
- Collecting a larger and more diverse dataset to improve the model's generalization.
- Implementing advanced perception and control algorithms for handling complex driving scenarios.
- Integrating real-time object detection and tracking for enhanced safety and situational awareness.

