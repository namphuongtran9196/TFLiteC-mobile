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

    public static Bitmap drawResult(Bitmap bitmap, float[] results){
        if (results == null || results.length <=0){
            return bitmap;
        }

        Bitmap temp = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(temp);
        Paint paint = new Paint();
        paint.setColor(Color.GREEN);
        paint.setStrokeWidth(10f);
        paint.setStyle(Paint.Style.STROKE);
        for (int i=0; i < results.length; i+=5){
            float score = results[i+4];
            if (score > 0.5) {
                float x_min = results[i + 0];
                float y_min = results[i + 1];
                float x_max = results[i + 2];
                float y_max = results[i + 3];

                canvas.drawLine(x_min, y_min, x_max, y_min, paint);
                canvas.drawLine(x_max, y_min, x_max, y_max, paint);
                canvas.drawLine(x_max, y_max, x_min, y_max, paint);
                canvas.drawLine(x_min, y_min, x_min, y_max, paint);
            } else{
                break;
            }
        }
        return temp;
    }
}