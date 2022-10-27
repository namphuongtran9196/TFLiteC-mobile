package vn.nam.tflite;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.content.res.AssetManager;

import vn.nam.tflite.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'tflite' library on application startup.
    static {
        System.loadLibrary("tfliteC");
    }

    private Button btnDetect;
    private ImageView imvImage;
    private ProgressBar pgrWait;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnDetect = findViewById(R.id.btnDetect);
        imvImage = findViewById(R.id.imvImage);
        pgrWait = findViewById(R.id.pgrWait);
        btnDetect.setOnClickListener(view -> {
            callDetect();
        });

        // init tflite model
        TensorUtils.modelAddr = TensorUtils.initTFliteModel(this.getAssets());
    }

    private void callDetect(){

        showView(true);
        new Thread(() -> {
            Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.test);
            float[] result = TensorUtils.detectC(TensorUtils.modelAddr, bitmap);
            Log.i("test", " " + result[0]);
            Log.i("test", " " + result[1]);
            Log.i("test", " " + result[2]);
            Log.i("test", " " + result[3]);
            Log.i("test", " " + result[4]);

//            // the results of process
//            float[][] results = new float[TensorUtils.VECTOR_OUT_SIZE_POST][8];
//
//            // maximum faces for detecting is 200 faces x 4 bbox points + score face
//            float[] faces_bbox = new float[TensorUtils.VECTOR_OUT_SIZE_POST* 5];
//            int valid_predict_index = 0;
//            int num_faces = 0;
//
//            // detect and return flatten array 1200 from (200x6) | xmin, ymin, xmax, ymax, head score, face score
//            float[] bboxes_raw = tensorUtils.detectHeadFace(bitmap);
//            if (bboxes_raw != null){
//                // filter face
//                for (int i = 0; i < TensorUtils.VECTOR_OUT_SIZE_POST * TensorUtils.VECTOR_OUT_WEIGHT_POST; i+=TensorUtils.VECTOR_OUT_WEIGHT_POST){
//                    float score_head = bboxes_raw[i + 4];
//                    float score_face = bboxes_raw[i + 5];
//                    if (score_head > TensorUtils.THRESH_HOLD_DETECTION || score_face > TensorUtils.THRESH_HOLD_DETECTION) {
//                        if (score_head > score_face) {
//                            for (int j = 0; j < 4; j++) {
//                                results[valid_predict_index][j] = bboxes_raw[i + j];
//                            }
//                            results[valid_predict_index][4] = 0.f; // class 0 for head
//                            results[valid_predict_index][5] = score_head; // score of head
//                            results[valid_predict_index][6] = -1.f; // dummy predict
//                            results[valid_predict_index][7] = -1.f; // dummy score
//                            valid_predict_index += 1;
//                        } else {
//                            for (int j = 0; j < 4; j++) {
//                                faces_bbox[num_faces * 5 + j] = bboxes_raw[i + j];
//                            }
//                            faces_bbox[num_faces * 5 + 4] = score_face;
//                            num_faces += 1;
//                        }
//                    }
//
//                }
//                // predict face features
//                float[] embs = tensorUtils.predictEmb(bitmap, faces_bbox, num_faces, true);
//
//                // perform recognition process
//                float[] preds = tensorUtils.predictID(embs, embsDatabase, num_faces, numEmbsInDatabase);
//
//                for (int i = 0; i< num_faces; i ++){
//                    for (int j = 0; j < 4; j++) {
//                        results[valid_predict_index][j] = faces_bbox[i * 5 + j];
//                    }
//                    results[valid_predict_index][4] = 1.f; // class 1 for face
//                    results[valid_predict_index][5] = faces_bbox[i * 5 + 4]; // score of head
//
//                    results[valid_predict_index][6] = preds[i*2]; // id index predict
//                    results[valid_predict_index][7] = preds[i*2+1]; // id score predict
//                    Log.d("Nam", "[INFO] Cannot detect" + preds[i*2] + " " +  preds[i*2+1]);
//                    valid_predict_index += 1;
//                }
//                Bitmap bmResult = BitmapUtils.drawHeadFace(bitmap, results,valid_predict_index);
//                setImageResult(bitmap);
//            }
//            else {
//                Log.d("Nam", "[INFO] Cannot detect");
//            }
            showView(false);
        }).start();
    }
    private void showView(boolean isWait){
        runOnUiThread(() -> {
            btnDetect.setEnabled(!isWait);
            pgrWait.setVisibility(isWait? View.VISIBLE:View.GONE);
        });
    }

    private void setImageResult(Bitmap bitmap){
        runOnUiThread(()->{
            try {
                Bitmap scaleBM = BitmapUtils.formatBitmapIn(bitmap);
                imvImage.setImageBitmap(scaleBM);
            }catch (Exception e){
                // file too large
                e.printStackTrace();
            }
        });
    }

}