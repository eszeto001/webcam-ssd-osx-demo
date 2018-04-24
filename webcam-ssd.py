# Hacked code from original python notebook export
# object_detection_tutorial.ipynb
#
# CUSTOMIZE ME: Point to location with Tensorflow models directory
MODELS_DIR = "/Users/eszeto/test/tf/models"

import sys, os
from datetime import datetime
import cv2
import numpy as np
import tensorflow as tf

if tf.__version__ < '1.4.0':
  raise ImportError(
    'Please upgrade your tensorflow installation to v1.4.* or later!'
  )


RESEARCH_DIR = MODELS_DIR+"/research"
OBJECT_DETECTION_DIR = RESEARCH_DIR + "/object_detection"
sys.path.append(OBJECT_DETECTION_DIR)
sys.path.append(RESEARCH_DIR)
sys.path.append(MODELS_DIR)

# import local modules
from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_CKPT = "./%s/frozen_inference_graph.pb" % MODEL_NAME
PATH_TO_LABELS = OBJECT_DETECTION_DIR + "/data/mscoco_label_map.pbtxt"

NUM_CLASSES = 90


print("Download frozen model file ...")
sys.stdout.flush()
import six.moves.urllib as urllib
import tarfile
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
MODEL_FILE = MODEL_NAME + ".tar.gz"
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

print("Load model into memory")
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


print("Load class map data")
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Main loop in CAM.
# Incorporated original run_inference_for_single_image(),
# but made more efficient use of code inside loop.
print("="*70)
print("Click on display window to select.")
print("Type 'q' in cam display window to quit.")
print("="*70)
sys.stdout.flush()
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:

      # HACK - Have this outside the loop to run faster
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes']:
          tensor_name = key + ':0'
          if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
               tensor_name)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Webcam loop
      cap = cv2.VideoCapture(0)
      while cap.isOpened():
          ret, frame = cap.read()
          if not ret: break
          image_np = frame

          # Run the inference and massage the output
          output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: np.expand_dims(image_np, 0)})

          # all outputs are float32 numpy arrays, so convert types as appropriate
          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict[
             'detection_classes'][0].astype(np.uint8)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]

          #output_dict = run_inference_for_single_image2(image_np, sess)
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
             image_np,
             output_dict['detection_boxes'],
             output_dict['detection_classes'],
             output_dict['detection_scores'],
             category_index,
             use_normalized_coordinates=True,
             line_thickness=8)

          cv2.imshow("frame", image_np)
          if (cv2.waitKey(1) & 0xFF) == ord('q'):
             print("Quitting,")
             break
      cap.release()
      cv2.destroyAllWindows()

