@WuXinyang Try the following:
./mo_tf.py --input_model=<path_to_frozen.pb> --tensorflow_use_custom_operations_config extensions/front/tf/ssd_support.json --output="detection_boxes,detection_scores,num_detections"


@alex_z For usage on NCS, I need to add a flag in your code: --data_type FP16
since the MYRIAD plugin does not support FP32 and the converted models are FP32 by default.
Again thanks for your intuitions! I have been searching for long time to try to find way to make tensorflow object-detection model run on NCS!


I have got about 10 FPS with the ssd_mobilenet_v1_coco model.


What archive file have you used for model training? I use ssd_mobilenet_v1_coco_2017_11_17.tar.gz and TF 1.4.
(create a new frozen graph with tf 1.4)


~/Downloads/frozen_inference_graph.pb



sudo python3 ./mo_tf.py --input_meta_graph /home/ncs/Downloads/ssd_mobilenet_v1_coco_2018_01_28/model.ckpt.meta --tensorflow_use_custom_operations_config extensions/front/tf/ssd_support.json --tensorflow_object_detection_api_pipeline_config /home/ncs/Downloads/ssd_mobilenet_v1_coco_2018_01_28/pipeline.config



-------------------
AndroidCon related codes:


python object_detection/export_tflite_ssd_graph.py --pipeline_config_path=/home/ncs/Downloads/ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config --trained_checkpoint_prefix=/home/ncs/Downloads/ssdlite_mobilenet_v2_coco_2018_05_09/model.ckpt --output_directory=/home/ncs/Desktop --add_postprocessing_op=true

bazel run --config=opt tensorflow/contrib/lite/toco:toco -- --input_file=/home/ncs/Desktop/tflite_graph.pb --output_file=/home/ncs/Desktop/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_values=128 --change_concat_input_ranges=false --allow_custom_ops

bazel run --config=opt tensorflow/contrib/lite/toco:toco -- --input_file=/home/ncs/Desktop/tflite_graph.pb --output_file=/home/ncs/Desktop/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=FLOAT --allow_custom_ops

bazel build -c opt --config=android_arm{,64} --cxxopt='--std=c++11' "//tensorflow/contrib/lite/examples/android:tflite_demo"

bazel build -c opt --config=android_arm{,64} --cxxopt='--std=c++11' //tensorflow/contrib/lite/examples/android:tflite_demo

bazel build -c opt --cxxopt='--std=c++11' --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a //tensorflow/contrib/lite/examples/android:tflite_demo

ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18   0 byte for quantized
ssd_mobilenet_v2_coco_2018_03_29			    				not 0 byte for float
ssd_mobilenet_v2_quantized_300x300_coco_2018_09_14		0 byte for quantized	not 0 byte for float
ssdlite_mobilenet_v2_coco_2018_05_09							not 0 byte for float
ssd_inception_v2_coco_2018_01_28							not 0 byte for float
adb install bazel-bin/tensorflow/contrib/lite/examples/android/tflite_demo.apk


codes of the running app:

bazel run -c opt tensorflow/contrib/lite/toco:toco -- --input_file=$OUTPUT_DIR/tflite_graph.pb --output_file=$OUTPUT_DIR/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_values=128 --change_concat_input_ranges=false --allow_custom_ops --default_ranges_min=0 --default_ranges_max=6


bazel build -c opt --config=android_arm --cxxopt='--std=c++11' //tensorflow/contrib/lite/examples/android:tflite_demo
adb install -r bazel-bin/tensorflow/contrib/lite/examples/android/tflite_demo.apk

Notes:

* There is a problem with quantization of the model that can potentially be solved. However, we can proceed with ssdlite_mobilenet_v2 and ssd_inception_v2 (float version).
* Subo phone is a 32 bit. (remove {, 64})


ToDo:
1. The problem now is with the quantization. Try to run "bazel run" with float accuracy in the demo.(Try to first make the demo work all the way to the end!)
2. Try to make docker work.


