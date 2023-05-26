from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

class FeatureExtractor:
    """
    A class for extracting deep features from input images using the VGG16 model.

    References:
    - VGG16 Model Documentation: https://keras.io/api/applications/vgg/#vgg16-function
    """

    def __init__(self):
        """
        Initialize the FeatureExtractor with the VGG16 model.
        The model is configured to extract features from an input image by obtaining the output of the 'fc1' layer,
        which represents the raw features of the image. We don't care about the prediction, just the raw output
        from the first fully connected layer of VGG16. 
        
        Those raw feature extracted from fc1 will be stored later on after normalization. 
        """
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img):
        """
        Extract a deep feature from an input image.
        We're getting this from fc1 layer, our first fully connected layer in VGG16.
        Args:
            img: An image in PIL.Image format or loaded using tensorflow.keras.preprocessing.image.load_img.

        Returns:
            feature (np.ndarray): A deep feature with shape (4096, ).

        References:
        - PIL.Image: https://pillow.readthedocs.io/en/stable/reference/Image.html
        - tensorflow.keras.preprocessing.image.load_img: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/load_img
        - VGG16 Model Documentation: https://keras.io/api/applications/vgg/#vgg16-function
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

        # Extract deep features using the VGG16 model
        feature = self.model.predict(x)[0]

        # Normalize the feature vector
        feature = feature / np.linalg.norm(feature)

        return feature

