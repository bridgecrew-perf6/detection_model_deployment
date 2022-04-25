from django.shortcuts import render
from django.http import HttpResponse
import base64
from django.http import HttpResponseRedirect, HttpResponse, JsonResponse
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils






# Loading ResNet
GLOBAL_PATH = 'traitement_app/'
PATH_TO_MODEL_DIR_2 = GLOBAL_PATH + 'exported-models/my_faster_rcnn'
PATH_TO_SAVED_MODEL_2 = PATH_TO_MODEL_DIR_2 + "/saved_model"

# Load saved model and build the detection function
model_2 = tf.saved_model.load(PATH_TO_SAVED_MODEL_2)
detect_fn_2 = model_2.signatures['serving_default']


PATH_TO_LABELS = GLOBAL_PATH + 'annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def index(request):
    if request.method == 'GET':
        return render(request, 'traitement_app/index.html')
    else:
        if request.is_ajax():
            img_data = request.POST['img_data']
            dec = base64.decodebytes(bytes(img_data, 'utf-8'))
            with open("traitement_app/static/traitement_app/images/image.png", "wb") as fic:
                fic.write(dec)
            return JsonResponse({'message': 'operation réussi'}, status=200)

def image(request):
    return render(request, 'traitement_app/images.html')

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))



def ResNet101(request):
    image_path = 'traitement_app/static/traitement_app/images/image.png'

    image_np = load_image_into_numpy_array(image_path)
    
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn_2(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.50,
          agnostic_mode=False)

    plt.figure(figsize=(10, 10), dpi = 80)
    plt.axis('off')
    plt.imshow(image_np_with_detections) 
    plt.savefig('traitement_app/static/traitement_app/images/image.png')
    return render(request, 'traitement_app/images.html')