#include "tflite_cpu.h"

TFLiteModel::TFLiteModel(const char *model, long modelSize, bool quantized) {
	m_modelQuantized = quantized;
	if (modelSize > 0) {
		initDetectionModel(model, modelSize);
	}
}

TFLiteModel::~TFLiteModel() {
	if (m_modelBytes != nullptr) {
		free(m_modelBytes);
		m_modelBytes = nullptr;
	}

	m_hasDetectionModel = false;
}

// Credit: https://github.com/YijinLiu/tf-cpu/blob/master/benchmark/obj_detect_lite.cc
void TFLiteModel::initDetectionModel(const char *tfliteModel, long modelSize) {
	if (modelSize < 1) { return; }

	// Copy to model bytes as the caller might release this memory while we need it (EXC_BAD_ACCESS error on ios)
	m_modelBytes = (char *) malloc(sizeof(char) * modelSize);
	memcpy(m_modelBytes, tfliteModel, sizeof(char) * modelSize);
	m_model = tflite::FlatBufferModel::BuildFromBuffer(m_modelBytes, modelSize);

	if (m_model == nullptr) {
		printf("Failed to load model");
		return;
	}

	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	tflite::InterpreterBuilder builder(*m_model, resolver);
	builder(&m_interpreter);
	if (m_interpreter == nullptr) {
		printf("Failed to create interpreter");
		return;
	}

	// Allocate tensor buffers.
	if (m_interpreter->AllocateTensors() != kTfLiteOk) {
		printf("Failed to allocate tensors!");
		return;
	}

	m_interpreter->SetNumThreads(1);

	// Find input tensors.
	if (m_interpreter->inputs().size() != 1) {
		printf("Detection model graph needs to have 1 and only 1 input!");
		return;
	}

	m_input_tensor = m_interpreter->tensor(m_interpreter->inputs()[0]);
	if (m_modelQuantized && m_input_tensor->type != kTfLiteUInt8) {
		printf("Model input should be kTfLiteUInt8!");
		return;
	}

	if (!m_modelQuantized && m_input_tensor->type != kTfLiteFloat32) {
		printf("Model input should be kTfLiteFloat32!");
		return;
	}

	// Find output tensors.
	m_output_tensor = m_interpreter->tensor(m_interpreter->outputs()[0]);

	m_hasDetectionModel = true;
}

PredictResult *TFLiteModel::detect(Mat src) {
	PredictResult res[MAX_OUTPUT];
	if (!m_hasDetectionModel) {
		return res;
	}

	if (m_modelQuantized) {
		// Copy image into input tensor
		uchar *dst = m_input_tensor->data.uint8;
		memcpy(dst, src.data,
			   sizeof(uchar) * INPUT_SIZE * INPUT_SIZE * INPUT_CHANNELS);
	} else {
		// Copy image into input tensor
		float *dst = m_input_tensor->data.f;
		memcpy(dst, src.data,
			   sizeof(float) * INPUT_SIZE * INPUT_SIZE * INPUT_CHANNELS);
	}

	if (m_interpreter->Invoke() != kTfLiteOk) {
		printf("Error invoking detection model");
		return res;
	}

	const float *output = m_output_tensor->data.f;

	for (int i = 0; i < MAX_OUTPUT; ++i) {
		res[i].xmin = output[i*5];
		res[i].ymin = output[i*5+1];
		res[i].xmax = output[i*5+2];
		res[i].ymax = output[i*5+3];
		res[i].score_person = output[i*5+4];
	}

	return res;
}
