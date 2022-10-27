#include "tflite_cpu.h"

TFLiteModel::TFLiteModel(const char *model, long modelSize, bool quantized) {
	m_modelQuantized = quantized;
	if (modelSize > 0) {
		initDetectionModel(model, modelSize);
	}
}

void TFLiteModel::initDetectionModel(const char *tfliteModel, long modelSize) {

	// Copy to model bytes as the caller might release this memory while we need it (EXC_BAD_ACCESS error on ios)
	m_modelBytes = (char *) malloc(sizeof(char) * modelSize);
	memcpy(m_modelBytes, tfliteModel, sizeof(char) * modelSize);
	m_model = tflite::FlatBufferModel::BuildFromBuffer(m_modelBytes, modelSize);
	assert(m_model != nullptr);

	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	tflite::InterpreterBuilder builder(*m_model, resolver);
	builder(&m_interpreter);
	assert(m_interpreter != nullptr);

	// Allocate tensor buffers.
	assert(m_interpreter->AllocateTensors() == kTfLiteOk);
	assert(m_interpreter->Invoke() == kTfLiteOk);

	m_interpreter->SetNumThreads(1);
}

PredictResult *TFLiteModel::detect(Mat input) {
	PredictResult res[MAX_OUTPUT];

    // convert the input image to float32
    input.convertTo(input, CV_32FC3);

	// allocate the tflite tensor
	m_interpreter->AllocateTensors();

	// get input & output layer of tflite model
	float *inputLayer = m_interpreter->typed_input_tensor<float>(0);
	float *outputLayer = m_interpreter->typed_output_tensor<float>(0);

	float *input_ptr = input.ptr<float>(0);

	// copy the input image to input layer
	memcpy(inputLayer, input_ptr, input.size().width * input.size().height * input.channels() * sizeof(float));

	// compute model instance
	if (m_interpreter->Invoke() != kTfLiteOk) {
		printf("Error invoking detection model");
		return res;
	}

	for (int i = 0; i < MAX_OUTPUT; ++i) {
		res[i].xmin = outputLayer[i*5];
		res[i].ymin = outputLayer[i*5+1];
		res[i].xmax = outputLayer[i*5+2];
		res[i].ymax = outputLayer[i*5+3];
		res[i].score_person = outputLayer[i*5+4];
	}

	return res;
}
