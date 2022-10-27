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
            bitmap = BitmapUtils.drawResult(bitmap,result);
            setImageResult(bitmap);
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