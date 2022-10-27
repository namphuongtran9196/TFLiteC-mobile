package vn.nam.tflite;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class TensorUtils {
    public static long modelAddr;
    // init tflite model in C++
    public static native long initTFliteModel(AssetManager assetManager);
    // inference in C++
    public static native float[] detectC(long modelAddr, Bitmap input);
}
