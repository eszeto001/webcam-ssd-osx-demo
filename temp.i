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
