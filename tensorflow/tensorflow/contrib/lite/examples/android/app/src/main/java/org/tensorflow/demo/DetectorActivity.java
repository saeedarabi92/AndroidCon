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
import android.content.pm.PackageManager;
import android.widget.Toast;
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
import android.Manifest;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
//import android.os.Environment;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.telephony.SmsManager;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.File;
import java.io.FileOutputStream;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;
import java.util.HashMap;
import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.tracking.MultiBoxTracker;
import org.tensorflow.lite.demo.R; // Explicit import needed for internal Google builds.

import static android.Manifest.permission.SEND_SMS;
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
  private long clockCount = 0;
  private long startClock = 0;
  private long endClock = 0;
  //private long pictureCounterForSaving = 0;
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

  private final String outputFolderPath = "/sdcard/DCIM/TFoutput";
  private final String inputFolderPath = "/sdcard/DCIM/Camera";

  private HashMap<String, Double> TitleConfidence = new HashMap<>();
  private String user_number;
  private static final int MY_PERMISSION_REQUEST_SEND_SMS = 1;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    openPhotos();
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
    if (isFromFolder) {
      sensorOrientation = 0;
    }
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
  }

  OverlayView trackingOverlay;

  @Override
  protected void processImage() {
    endClock = SystemClock.uptimeMillis();
    clockCount = endClock - startClock;
    if (clockCount == 1000) {
      System.out.println("Pictures per second "+pictureCounter);
      startClock = endClock;
      clockCount = 0;
    }
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

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);

            ++pictureCounter;
            final long startTime = SystemClock.uptimeMillis();
            List<Classifier.Recognition> results = null;
            if (isFromFolder) {
//              System.out.println("Run on picture from folder!");
              results = detector.recognizeImage(croppedFromFolder);
            } else {
              results = detector.recognizeImage(croppedBitmap);
            }
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            LOGGER.i("This is the " + results);

            LOGGER.i("Last processing time ms " + lastProcessingTimeMs);
            // Write to file
            try {
              writeToFile(results);
            } catch (IOException e) {
              LOGGER.i("Error in writing result to the file.");
            }

            // calculate the total time for getting the average time
            totalTimeMs += lastProcessingTimeMs;
            LOGGER.i("The total average time ms  " + totalTimeMs/pictureCounter);

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

            // Result store
            HashMap<String, Vector<Double>> tempRes = new HashMap<>();

            for (final Classifier.Recognition result : results) {
              // For detection with so small confidence, we skip it when calculating the final confidence
              if (result.getConfidence() < 0.1) {continue;}
              String curTitle = result.getTitle();
              String titleConfidence = Double.toString(result.getConfidence());
              System.out.println(curTitle + " "+titleConfidence);
              Vector curInfo = null;

              if (!tempRes.containsKey(curTitle)) {
                tempRes.put(curTitle, new Vector<Double>());
                curInfo = tempRes.get(curTitle);
                curInfo.add(0.0);
                curInfo.add(0.0);
              } else {
                curInfo = tempRes.get(curTitle);
              }

              double curConfidence = doubleAdd(curInfo.get(0).toString(), titleConfidence);
              double curCounter = doubleAdd(curInfo.get(1).toString(), "1.0");
              curInfo.set(0, curConfidence);
              curInfo.set(1, curCounter);

              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
		

                canvas.drawRect(location, paint);
                if (isFromFolder) {
                  String res = result.getId() + " " + curTitle + " " + String.valueOf(curConfidence);

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

            for (String key : tempRes.keySet()) {
              double pictureConfidence = doubleDivide(tempRes.get(key).get(0).toString(), tempRes.get(key).get(1).toString());
              if (!TitleConfidence.containsKey(key)) {
                TitleConfidence.put(key, 0.0);
              }
              // here the pictureConfidence is just a sum, later should be divied by the number of pictures
              pictureConfidence = doubleAdd(Double.toString(pictureConfidence), TitleConfidence.get(key).toString());
              TitleConfidence.put(key, pictureConfidence/pictureCounter);
              // Threshold needs to be changed
              if (pictureConfidence/pictureCounter > 0.5) {
                System.out.println(key+" "+pictureConfidence/pictureCounter);
                sendSMS(key);
              }
//              System.out.println("Summary for Picture " + pictureCounter + " " + key+" "+TitleConfidence.get(key));
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

  public void writeToFile(List<Classifier.Recognition> results) throws IOException {
    String filename = getOutputFileName();

    String data = "";
    for (Classifier.Recognition result : results) {
      data += result.getTitle() + " " + result.getConfidence() + " " + getCoordinates(result.getLocation()) + "\n";
    }
    data = data.trim();

    String fileName = outputFolderPath + File.separator + filename + ".txt";

    FileOutputStream outputFile = new FileOutputStream(fileName);
    outputFile.write(data.getBytes());
    outputFile.close();
  }

  private String getOutputFileName () {
    String filename = "";
    if (isFromFolder) {
      String originalName = allPicture[counterForFolderPicture-1].getName();
      int index = originalName.indexOf(".");
      filename = originalName.substring(0, index);
    } else {
      filename = String.valueOf(pictureCounter);
    }
    return filename;
  }

  public void sendSMS(String result) {
    SmsManager sm = SmsManager.getDefault();
    String sms = result + " is detected!";

    try {
      sm.sendTextMessage(user_number,null,sms,null,null);
      Toast.makeText(getApplicationContext(), "SMS Sent Successfully!", Toast.LENGTH_SHORT).show();
    } catch (Exception e) {
      Toast.makeText(getApplicationContext(), "SMS Failed to Sent.", Toast.LENGTH_SHORT).show();
    }
  }


  public void openPhotos() {
    File folder = new File(inputFolderPath);
    allPicture = folder.listFiles();
    if (allPicture.length == 0) {
      isFromFolder = false;
    } else {
      isFromFolder = true;
    }
  }

  public void loadPicture() {
    File f = allPicture[counterForFolderPicture];
    String picture = inputFolderPath + File.separator + f.getName();
    Bitmap folderPicOrigin = BitmapFactory.decodeFile(picture);
    if (folderPicOrigin != null) {
      folderPicToCrop(folderPicOrigin);
    }
    counterForFolderPicture ++;
    if (counterForFolderPicture == allPicture.length) {
      folderPicRemain = false;
    }
  }

  public void saveToFile(Bitmap croppedBitmap) {
    String filename = getOutputFileName();

    File outFile = new File(outputFolderPath, filename+"_output.jpg");
    try {
      FileOutputStream out = new FileOutputStream(outFile);
      croppedBitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
//      System.out.println("cropped saved !");
    } catch (Exception e) {
      LOGGER.e("Cropped not saved");
    }
  }

  public void folderPicToCrop(Bitmap bitmap) {
    final Canvas toCrop = new Canvas(croppedFromFolder);
    toCrop.drawBitmap(bitmap, frameToCropTransform, null);
  }

  private static double doubleAdd(String v1, String v2) {
    BigDecimal b1 = new BigDecimal(v1);
    BigDecimal b2 = new BigDecimal(v2);

    return b1.add(b2).doubleValue();
  }

  private static double doubleDivide(String v1, String v2) {
    BigDecimal b1 = new BigDecimal(v1);
    BigDecimal b2 = new BigDecimal(v2);

    return b1.divide(b2, 4, RoundingMode.HALF_UP).doubleValue();
  }

  private static String getCoordinates(RectF location) {
    String raw = location.toString();
    raw = raw.substring(6, raw.length()-1);
    String[] temp = raw.split(",");
    String ans = "";
    for (String s : temp) {
      ans = ans + s + " ";
    }
    return ans.trim();
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    Intent intent = getIntent();
    String number = intent.getStringExtra(MainActivity.EXTRA_MESSAGE);
    user_number = number;
    Toast.makeText(getApplicationContext(), "The number is "+number, Toast.LENGTH_LONG).show();

    int permissionCode = 1;
    String[] permission = {Manifest.permission.SEND_SMS};
    if (ActivityCompat.checkSelfPermission(this, permission[0]) != PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(this, permission, permissionCode);
    }
  }


}
