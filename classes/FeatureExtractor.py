# The FeatureExtractor class uses the ResNet50 model to extract deep features from input images.
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.models import Model
import numpy as np
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
class FeatureExtractor:
    """
    A class for extracting deep features from input images using the ResNet50 model.

    References:
    - ResNet50 Model Documentation: https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50
    """

    def __init__(self, gpu_mode = False):
        """
        Initialize the FeatureExtractor with the ResNet50 model.
        The model is configured to extract features from an input image by obtaining the output of the 'avg_pool' layer,
        which represents the raw features of the image. 
     
        """
        if (gpu_mode):
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                    tf.config.experimental.set_memory_growth(gpus[0], True)
                    print("GPU enabled:", tf.config.list_physical_devices('GPU'))
                except RuntimeError as e:
                    print(e)
            else:
                print("GPU is not available. Using CPU.")
        
        base_model = ResNet50(weights='imagenet')
         # Load the ResNet50 model without the top (classification) layers
        
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        self.prediction_model = base_model
        self.gpu_mode = gpu_mode
        

    def extract(self, img):
        """
        Extract a deep feature from an input image.
        We're getting this from avg_pool, our best layer for feature representation ResNet50.
        Args:
            img: An image in PIL.Image format or loaded using tensorflow.keras.preprocessing.image.load_img.

        Returns:
            feature (np.ndarray): A deep feature with shape (4096, ).

        References:
        - PIL.Image: https://pillow.readthedocs.io/en/stable/reference/Image.html
        - tensorflow.keras.preprocessing.image.load_img: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/load_img
        - ResNet50 Model Documentation: https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50
        """
        
        # Resize the image to match the input size of VGG16 (224x224)
        img = img.resize((224, 224))

        # Convert the image to RGB color space
        img = img.convert('RGB')

        # Convert the image to a numpy array (Height x Width x Channel)
        x = image.img_to_array(img)

        # Expand dimensions to match the model input shape (1, H, W, C)
        x = np.expand_dims(x, axis=0)

        # Preprocess the input by subtracting average values for each pixel
        x = preprocess_input(x)
        
        if (self.gpu_mode):
            with tf.device('/GPU:0'):
                feature = self.model.predict(x)[0]
        else:
            feature = self.model.predict(x)[0]

        # Extract deep features using the VGG16 model
        # feature = self.model.predict(x)[0]

        # Normalize the feature vector
        feature = feature / np.linalg.norm(feature)

        return feature

    def predict(self, img):
            """
            Perform image classification on the input image using the ResNet50 model.
            Args:
                img: An image in PIL.Image format or loaded using tensorflow.keras.preprocessing.image.load_img.

            Returns:
                predictions (list): A list of (class_label, probability) tuples representing the top predicted classes.
            """
            # Resize the image to match the input size of ResNet50 (224x224)
            img = img.resize((224, 224))

            # Convert the image to RGB color space
            img = img.convert('RGB')

            # Convert the image to a numpy array (Height x Width x Channel)
            x = image.img_to_array(img)

            # Expand dimensions to match the model input shape (1, H, W, C)
            x = np.expand_dims(x, axis=0)

            # Preprocess the input by subtracting average values for each pixel
            x = preprocess_input(x)

            if self.gpu_mode:
                with tf.device('/GPU:0'):
                    predictions = self.prediction_model.predict(x)
            else:
                predictions = self.prediction_model.predict(x)

            # Decode the predictions to obtain the ImageNet class labels
            decoded_predictions = decode_predictions(predictions, top=5)[0]

            return decoded_predictions