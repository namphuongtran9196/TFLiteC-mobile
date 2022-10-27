#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

using namespace cv;

struct PredictResult {
	float score_person = 0;
	float ymin = 0.0;
	float xmin = 0.0;
	float ymax = 0.0;
	float xmax = 0.0;
};

class TFLiteModel {
public:
	TFLiteModel(const char *model, long modelSize, bool quantized = false);
    ~TFLiteModel();
    PredictResult *detect(Mat src);
    const int MAX_OUTPUT = 200;
private:
	// members
	const int INPUT_SIZE = 352;
	const int INPUT_CHANNELS = 3;
	bool m_modelQuantized = false;
	bool m_hasDetectionModel = false;
	char *m_modelBytes = nullptr;
	std::unique_ptr<tflite::FlatBufferModel> m_model;
	std::unique_ptr<tflite::Interpreter> m_interpreter;

	// Methods
	void initDetectionModel(const char *model, long modelSize);
};