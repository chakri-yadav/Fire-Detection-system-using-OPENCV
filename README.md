# Fire Detection with OpenCV and InceptionV3

This Python script uses OpenCV for capturing video frames and the InceptionV3 model for fire detection in the video stream. When fire is detected, it displays the frame in grayscale, and you can quit the program by pressing 'q'.

## Getting Started

### Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- TensorFlow
- Keras

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/fire-detection.git
   ```

2. Install the required dependencies:

   ```bash
   pip install opencv-python tensorflow keras
   ```

### Usage

1. Run the Python script to start fire detection in the live video feed.

   ```bash
   python fire_detection.py
   ```

2. To exit the program, press 'q' in the OpenCV window.

### Customization

- You can customize the model used for fire detection by replacing `InceptionV3.h5` with another pre-trained model.
- Modify the code to suit your specific video processing and fire detection requirements.

## Code Structure

The code is organized as follows:

- **`fire_detection.py`**: The main Python script for fire detection using OpenCV and the InceptionV3 model.
- **`README.md`**: This documentation file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Credits to the developers of the InceptionV3 model.
- Thanks to the OpenCV, TensorFlow, and Keras communities for their excellent libraries.

## About

This project is maintained by Chakravarthi Nukala. For questions or contributions, please contact chakravarthinukala@gmail.com.

