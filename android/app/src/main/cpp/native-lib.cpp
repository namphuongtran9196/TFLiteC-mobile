#include <jni.h>
#include "tflite_cpu.h"
#include <android/log.h>
#include "android/bitmap.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

// Convert the bitmap to OpenCV Mat
// Reference https://github.com/opencv/opencv/blob/17234f82d025e3bbfbf611089637e5aa2038e7b8/modules/java/generator/src/cpp/utils.cpp
void bitmapToMat(JNIEnv * env, jobject bitmap, Mat& dst, jboolean needUnPremultiplyAlpha)
{
    AndroidBitmapInfo  info;
    void*              pixels = 0;

    try {
        CV_Assert( AndroidBitmap_getInfo(env, bitmap, &info) >= 0 );
        CV_Assert( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                   info.format == ANDROID_BITMAP_FORMAT_RGB_565 );
        CV_Assert( AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0 );
        CV_Assert( pixels );
        dst.create(info.height, info.width, CV_8UC4);
        if( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 )
        {
            Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if(needUnPremultiplyAlpha) cvtColor(tmp, dst, COLOR_mRGBA2RGBA);
            else tmp.copyTo(dst);
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            Mat tmp(info.height, info.width, CV_8UC2, pixels);
            cvtColor(tmp, dst, COLOR_BGR5652RGBA);
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch(const cv::Exception& e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nBitmapToMat}");
        return;
    }
}

// Convert a mat to Bitmap
// Reference https://github.com/opencv/opencv/blob/17234f82d025e3bbfbf611089637e5aa2038e7b8/modules/java/generator/src/cpp/utils.cpp
void matToBitmap(JNIEnv * env, Mat src, jobject bitmap, jboolean needPremultiplyAlpha)
{
    AndroidBitmapInfo  info;
    void*              pixels = 0;

    try {
        CV_Assert( AndroidBitmap_getInfo(env, bitmap, &info) >= 0 );
        CV_Assert( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                   info.format == ANDROID_BITMAP_FORMAT_RGB_565 );
        CV_Assert( src.dims == 2);
        CV_Assert(info.height == (uint32_t)src.rows);
        CV_Assert(info.width == (uint32_t)src.cols );
        CV_Assert( src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4 );
        CV_Assert( AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0 );
        CV_Assert( pixels );
        if( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 )
        {
            Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if(src.type() == CV_8UC1)
            {
                cvtColor(src, tmp, COLOR_GRAY2RGBA);
            } else if(src.type() == CV_8UC3){
                cvtColor(src, tmp, COLOR_RGB2RGBA);
            } else if(src.type() == CV_8UC4){
                if(needPremultiplyAlpha) cvtColor(src, tmp, COLOR_RGBA2mRGBA);
                else src.copyTo(tmp);
            }
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            Mat tmp(info.height, info.width, CV_8UC2, pixels);
            if(src.type() == CV_8UC1)
            {
                cvtColor(src, tmp, COLOR_GRAY2BGR565);
            } else if(src.type() == CV_8UC3){
                cvtColor(src, tmp, COLOR_RGB2BGR565);
            } else if(src.type() == CV_8UC4){
                cvtColor(src, tmp, COLOR_RGBA2BGR565);
            }
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch(const cv::Exception& e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nMatToBitmap}");
        return;
    }
}
extern "C"
JNIEXPORT jlong JNICALL
Java_vn_nam_tflite_TensorUtils_initTFliteModel(JNIEnv *env, jclass clazz, jobject assetManager) {
    char *buffer = nullptr;
    long size = 0;

    if (!(env->IsSameObject(assetManager, NULL))) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        AAsset *asset = AAssetManager_open(mgr, "yolo_fastest_v2.tflite", AASSET_MODE_UNKNOWN);
        assert(asset != nullptr);

        size = AAsset_getLength(asset);
        buffer = (char *) malloc(sizeof(char) * size);
        AAsset_read(asset, buffer, size);
        AAsset_close(asset);
    }

    jlong res = (jlong) new TFLiteModel(buffer, size, false);
    free(buffer); // ObjectDetector duplicate it and responsible to free it

    return res;
}
extern "C"
JNIEXPORT jfloatArray JNICALL
Java_vn_nam_tflite_TensorUtils_detectC(JNIEnv *env, jclass clazz, jlong modelAddr, jobject input) {
    Mat input_mat;
    // convert bitmap to OpenCV Mat
    bitmapToMat(env, input, input_mat, false);
    resize(input_mat, input_mat, Size(352, 352), 0, 0, INTER_LINEAR);
    // inference
    TFLiteModel *model = (TFLiteModel *) modelAddr;
    PredictResult *res = model->detect(input_mat);

    // Encode each detection as 6 numbers (label,score,xmin,xmax,ymin,ymax)
    int resArrLen = model->MAX_OUTPUT * 5;
    jfloat jres[resArrLen];
    for (int i = 0; i < model->MAX_OUTPUT; ++i) {
        jres[i * 5] = res[i].xmin;
        jres[i * 5 + 1] = res[i].ymin;
        jres[i * 5 + 2] = res[i].xmax;
        jres[i * 5 + 3] = res[i].ymax;
        jres[i * 5 + 4] = res[i].score_person;
    }

    jfloatArray detections = env->NewFloatArray(resArrLen);
    env->SetFloatArrayRegion(detections, 0, resArrLen, jres);

    return detections;
}