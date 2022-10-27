package vn.nam.tflite;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;

public class BitmapUtils {

    public static Bitmap formatBitmapIn(Bitmap bitmap) {
        if (bitmap != null && bitmap.getByteCount() > 0) {
            Bitmap bm = Bitmap.createScaledBitmap(bitmap,
                    640,
                    640, false);
            return bm.copy(Bitmap.Config.ARGB_8888, true);
        }
        return null;
    }

    public static Bitmap drawHeadFace(Bitmap bitmap, float[][] results, int numResults){
        if (results == null || results.length <=0){
            return bitmap;
        }

        Bitmap temp = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(temp);
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStrokeWidth(10f);
        paint.setStyle(Paint.Style.STROKE);
        for (int i=0; i < numResults; i++){
            float obj_class = results[i][4];
            float id_predict = results[i][6];
            if (obj_class == 0.0f){
                paint.setColor(Color.RED);
            }
            else {
                if (id_predict != -1f){
                    Log.d("Nam", "[INFO] predict score: " + results[i][7]);
                    if (id_predict == 0){
                        paint.setColor(Color.rgb(255,255,0));
                    }
                    else if (id_predict == 1){
                        paint.setColor(Color.rgb(0,255,255));
                    }
                    else{
                        paint.setColor(Color.GREEN);
                    }
                }
                else{
                    paint.setColor(Color.BLUE);
                }
            }


            float x_min = results[i][0];
            float y_min = results[i][1];
            float x_max = results[i][2];
            float y_max = results[i][3];

            canvas.drawLine(x_min, y_min, x_max, y_min, paint);
            canvas.drawLine(x_max, y_min, x_max, y_max, paint);
            canvas.drawLine(x_max, y_max, x_min, y_max, paint);
            canvas.drawLine(x_min, y_min, x_min, y_max, paint);
        }
        return temp;
    }
}