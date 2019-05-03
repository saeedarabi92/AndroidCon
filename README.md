# This repo contains all the info about AndroidCon project.

# To build the App:
1. clone the repo
2. from _tensorflow_ directory:
  run:
  `bazel build -c opt --config=android_arm --cxxopt='--std=c++11' //tensorflow/contrib/lite/examples/android:tflite_demo
  adb install -r bazel-bin/tensorflow/contrib/lite/examples/android/tflite_demo.apk`
