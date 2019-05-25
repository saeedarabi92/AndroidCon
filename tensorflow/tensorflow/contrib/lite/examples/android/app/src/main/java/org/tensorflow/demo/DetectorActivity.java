/*
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.net.Uri;
import android.os.SystemClock;
import android.os.Environment;
import android.telephony.SmsManager;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.File;
import java.io.FileOutputStream;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;
import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.tracking.MultiBoxTracker;
import org.tensorflow.lite.demo.R; // Explicit import needed for internal Google builds.

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/pets_labels_list.txt";
  
  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  private static final DetectorMode MODE = DetectorMode.TF_OD_API;

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;

  private static final boolean MAINTAIN_ASPECT = false;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;

  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;
  private long totalTimeMs = 0;
  private long pictureCounter = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private byte[] luminanceCopy;

  private BorderedText borderedText;
  // globals for files in folder
  private Bitmap croppedFromFolder = null;
  private boolean isFromFolder = false;
  private File[] allPicture = null;
  private int counterForFolderPicture = 0;
  private boolean folderPicRemain = true;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              getAssets(),
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      LOGGER.e("Exception initializing classifier!", e);
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();
    System.out.println("PreviewHeight "+previewHeight+", previewWidth "+previewWidth);
    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);
    croppedFromFolder = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            if (!isDebug()) {
              return;
            }
            final Bitmap copy = cropCopyBitmap;
            if (copy == null) {
              return;
            }

            final int backgroundColor = Color.argb(100, 0, 0, 0);
            canvas.drawColor(backgroundColor);

            final Matrix matrix = new Matrix();
            final float scaleFactor = 2;
            matrix.postScale(scaleFactor, scaleFactor);
            matrix.postTranslate(
                canvas.getWidth() - copy.getWidth() * scaleFactor,
                canvas.getHeight() - copy.getHeight() * scaleFactor);
            canvas.drawBitmap(copy, matrix, new Paint());

            final Vector<String> lines = new Vector<String>();
            if (detector != null) {
              final String statString = detector.getStatString();
              final String[] statLines = statString.split("\n");
              for (final String line : statLines) {
                LOGGER.w("what is line" + line);
                lines.add(line);
              }
            }
            lines.add("");

            lines.add("Frame: " + previewWidth + "x" + previewHeight);
            lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
            lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
            lines.add("Rotation: " + sensorOrientation);
            lines.add("Inference time: " + lastProcessingTimeMs + "ms");

            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
          }
        });

    // Test SMS, No SIM card yet
//    sendSMS("6262239634", "Hello Sally");
    // Test open picture
    openPhotos();
  }

  OverlayView trackingOverlay;

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    byte[] originalLuminance = getLuminance();
    tracker.onFrame(
        previewWidth,
        previewHeight,
        getLuminanceStride(),
        sensorOrientation,
        originalLuminance,
        timestamp);
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");
    
    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];
    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);

    if (isFromFolder) {
      loadPicture();
    } else {
      readyForNextImage();
    }

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }
    //saveToFile(croppedBitmap, true);

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            List<Classifier.Recognition> results = null;
            if (isFromFolder) {
              System.out.println("Run on picture from folder!");
              results = detector.recognizeImage(croppedFromFolder);
            } else {
              results = detector.recognizeImage(croppedBitmap);
            }
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            LOGGER.i("This is the " + results);

            LOGGER.i("Last processing time ms " + lastProcessingTimeMs);
            // Write to file
            try {
              writeToFile(results, currTimestamp);
            } catch (IOException e) {
              LOGGER.i("Error in writing result to the file.");
            }

            // calculate the total time for getting the average time
             totalTimeMs += lastProcessingTimeMs;
            LOGGER.i("The total processing time ms  " + totalTimeMs);

            Canvas canvas = null;
            if (isFromFolder) {
              cropCopyBitmap = Bitmap.createBitmap(croppedFromFolder);
              canvas = new Canvas(croppedFromFolder);
            } else {
              cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
              canvas = new Canvas(cropCopyBitmap);
            }

            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              System.out.println(result.toString());
              if (location != null && result.getConfidence() >= minimumConfidence) {
		

                canvas.drawRect(location, paint);
                if (isFromFolder) {
                  String res = result.getId() + " " + result.getTitle() + " " + String.valueOf(result.getConfidence());

                  Paint paintText = new Paint();
                  paintText.setColor(Color.RED);
                  paintText.setTextSize(15);
                  float x = location.left;
                  float y = location.bottom + 15;
                  canvas.drawText(res, x, y, paintText);
                }
                cropToFrameTransform.mapRect(location);
                result.setLocation(location);
                mappedRecognitions.add(result);
              }
            }

            tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
            trackingOverlay.postInvalidate();

            requestRender();
            computingDetection = false;
            if (isFromFolder) {
              saveToFile(croppedFromFolder);
              if (folderPicRemain){
                processImage();
              }
            }
          }
        });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onSetDebug(final boolean debug) {
    detector.enableStatLogging(debug);
  }

  public void writeToFile(List<Classifier.Recognition> results, long currTimestamp) throws IOException {
    String data = "";
    for (Classifier.Recognition result : results) {
      data += result.toString() + " ";
    }
    data = data.trim();

    Context context = getApplicationContext();
    File file = context.getFilesDir();
    String fileName = file.getAbsolutePath() + File.separator + String.valueOf(currTimestamp)+".txt";
    //System.out.println(fileName);
    FileOutputStream outputFile = new FileOutputStream(fileName);
    outputFile.write(data.getBytes());
    outputFile.close();


  }

  public void sendSMS(String phone, String result) {
    SmsManager sm = SmsManager.getDefault();
    List<String> smslist = sm.divideMessage(result);
    for (String sms : smslist) {
      sm.sendTextMessage(phone,null,sms,null,null);
    }
  }

  public void openPhotos() {
    String folderPath = "/sdcard/DCIM/Camera";
    File folder = new File(folderPath);
    allPicture = folder.listFiles();
    System.out.println("the number of pictures: "+allPicture.length);
    if (allPicture.length == 0) {
      isFromFolder = false;
    } else {
      isFromFolder = true;
    }
  }

  public void loadPicture() {
    String folderPath = "/sdcard/DCIM/Camera";
    File f = allPicture[counterForFolderPicture];
    String picture = folderPath + "/" + f.getName();
    Bitmap folderPicOrigin = BitmapFactory.decodeFile(picture);
    if (folderPicOrigin != null) {
      System.out.println("Folder pic process");
      folderPicToCrop(folderPicOrigin);
 //     saveToFile(croppedFromFolder);
    }
    counterForFolderPicture ++;
    if (counterForFolderPicture == allPicture.length) {
      folderPicRemain = false;
    }
  }

  public void saveToFile(Bitmap croppedBitmap) {
    ++pictureCounter;
    String folderPath = "/sdcard/DCIM/TFoutput";
//    if (!isCropped){
//      final Canvas canvas = new Canvas(croppedBitmap);
//      canvas.drawBitmap(bit, frameToCropTransform, null);
//      // For examining the actual TF input.
//      if (SAVE_PREVIEW_BITMAP) {
//        ImageUtils.saveBitmap(croppedBitmap);
//      }
//    }

    File outFile = new File(folderPath, pictureCounter+".jpg");
    try {
      FileOutputStream out = new FileOutputStream(outFile);
      croppedBitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
      System.out.println("cropped saved !");
    } catch (Exception e) {
      LOGGER.e("Cropped not saved");
    }
  }

  public void folderPicToCrop(Bitmap bitmap) {
    final Canvas toCrop = new Canvas(croppedFromFolder);
    toCrop.drawBitmap(bitmap, frameToCropTransform, null);
//    System.out.println("In folder bitmap H "+bitmap.getHeight()+", W "+bitmap.getWidth());
//    System.out.println("In folder cropped H "+croppedFromFolder.getHeight()+", W "+croppedFromFolder.getWidth());
  }
}
