# This repo contains all the info about AndroidCon project.

### To convert the .pb file to tflite graph, form _object_detection_ directory, run:

python object_detection/export_tflite_ssd_graph.py --pipeline_config_path=/home/ncs/Downloads/ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config --trained_checkpoint_prefix=/home/ncs/Downloads/ssdlite_mobilenet_v2_coco_2018_05_09/model.ckpt --output_directory=/home/ncs/Desktop --add_postprocessing_op=true

### To convert the tflite graph to TensorFlow Lite flatbuffer format , form _object_detection_ directory, run:

This will convert the resulting frozen graph (tflite_graph.pb) to the TensorFlow Lite flatbuffer format (detect.tflite) via the following command

* for quantized models:

`bazel run --config=opt tensorflow/contrib/lite/toco:toco -- --input_file=/home/ncs/Desktop/tflite_graph.pb --output_file=/home/ncs/Desktop/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_values=128 --change_concat_input_ranges=false --allow_custom_ops`

* for simple models (not quantized):

`bazel run --config=opt tensorflow/contrib/lite/toco:toco -- --input_file=/home/ncs/Desktop/tflite_graph.pb --output_file=/home/ncs/Desktop/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=FLOAT --allow_custom_ops`


### To build the App:
1. clone the repo
2. from _tensorflow_ directory:

  run:

  `bazel build -c opt --config=android_arm --cxxopt='--std=c++11' //tensorflow/contrib/lite/examples/android:tflite_demo
  adb install -r bazel-bin/tensorflow/contrib/lite/examples/android/tflite_demo.apk`

3. to install the App in the phone:
`adb install bazel-bin/tensorflow/contrib/lite/examples/android/tflite_demo.apk`


* ToDo:

Train the model with quantization.
