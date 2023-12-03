#include "nn_tensorrt.hpp"

using namespace nvonnxparser;
using namespace nvinfer1;

// https://wunkolo.github.io/post/2022/02/memory-size-literals/
constexpr std::size_t operator""_KiB(unsigned long long int x) {
	return 1024ULL * x;
}

constexpr std::size_t operator""_MiB(unsigned long long int x) {
	return 1024_KiB * x;
}

constexpr std::size_t operator""_GiB(unsigned long long int x) {
	return 1024_MiB * x;
}

constexpr std::size_t operator""_TiB(unsigned long long int x) {
	return 1024_GiB * x;
}

constexpr std::size_t operator""_PiB(unsigned long long int x) {
	return 1024_TiB * x;
}

class Logger : public ILogger
{
	void log(Severity severity, const char* msg) noexcept override
	{
		// suppress info-level messages
		if (severity <= Severity::kINTERNAL_ERROR)
			std::cout << "InError: " << msg << std::endl;
		if (severity <= Severity::kERROR)
			std::cout << "Error: " << msg << std::endl;
		else if (severity <= Severity::kWARNING)
			std::cout << "Warning: " << msg << std::endl;
	}
} logger;

void NNTensorRT::init(const std::string& onnx_file_name) {

	// ここで、次善にdevice memory を確保
#if defined (FEATURE_V2)
	cudaMalloc((void**)&(std::get<0>(_input_device)), SIZEOF_FEATURE1 * _max_batch_size);
	cudaMalloc((void**)&(std::get<1>(_input_device)), SIZEOF_FEATURE2 * _max_batch_size);
#if defined(USE_PACKED_FEATURE)
	cudaMalloc((void**)&(std::get<0>(_unpacked_input_device)), sizeof(float) * N_FEATURE1_WHC * _max_batch_size);
	cudaMalloc((void**)&(std::get<1>(_unpacked_input_device)), sizeof(float) * N_FEATURE2_WHC * _max_batch_size);
#endif

#else
	cudaMalloc((void**)&_input_device, sizeof(float) * N_FEATURE_WHC * _max_batch_size);
#endif

	cudaMalloc((void**)&_output_p_device, sizeof(float) * N_LABEL_SIZE * _max_batch_size);
	cudaMalloc((void**)&_output_v_device, sizeof(float) * 1 * _max_batch_size);

	// binding の一覧。
	// enqueue に渡してやると、idx = 0を入力、idx = 1を出力としてデータをやりくりしてくれるそうな。。。
#if defined (FEATURE_V2)
#if defined(USE_PACKED_FEATURE)
	_input_bindings = { std::get<0>(_unpacked_input_device), std::get<1>(_unpacked_input_device), _output_p_device, _output_v_device };
#else
	_input_bindings = { std::get<0>(_input_device), std::get<1>(_input_device), _output_p_device, _output_v_device };
#endif
#else
	_input_bindings = { _input_device, _output_p_device, _output_v_device };
#endif

	if (!load_model(onnx_file_name)) {
		std::cout << "info string Error: failed to load_model()" << std::endl;
		exit(1);
	}
}

NNTensorRT::NNTensorRT(const std::string& onnx_file_name, const int max_batch_size)
	: _max_batch_size(max_batch_size), _is_fp16(false)
{
	init(onnx_file_name);
}

NNTensorRT::NNTensorRT(const std::string& onnx_file_name, const int max_batch_size, const bool is_fp16)
	: _max_batch_size(max_batch_size), _is_fp16(is_fp16)
{
	init(onnx_file_name);
}

NNTensorRT::~NNTensorRT() {
#if defined (FEATURE_V2)
	cudaFree(std::get<0>(_input_device));
	cudaFree(std::get<1>(_input_device));
#if defined(USE_PACKED_FEATURE)
	cudaFree(std::get<0>(_unpacked_input_device));
	cudaFree(std::get<1>(_unpacked_input_device));
#endif

#else
	cudaFree(_input_device);
#endif

	cudaFree(_output_p_device);
	cudaFree(_output_v_device);
}

bool NNTensorRT::build_engine(const std::string& onnx_file_name) {
	//// build 環境の整備
	auto builder = InferUniquePtr<IBuilder>(createInferBuilder(logger));
	if (!builder) {
		return false;
	}

	// こいつ無しでのnetwork 定義は非推奨
	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = InferUniquePtr<INetworkDefinition>(builder->createNetworkV2(explicitBatch));
	if (!network) {
		std::cout << "info string Error!: failed to createNetworkV2()" << std::endl;
		return false;
	}

	auto config = InferUniquePtr<IBuilderConfig>(builder->createBuilderConfig());
	if (!config) {
		std::cout << "info string Error!: failed to createBuilderConfig()" << std::endl;
		return false;
	}

	auto parser = InferUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
	if (!parser) {
		std::cout << "info string Error!: failed to createParser()" << std::endl;
		return false;
	}

	auto parsed = parser->parseFromFile(onnx_file_name.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));
	if (!parsed) {
		std::cout << "info string Error: filed to parseFromFile()" << std::endl;
		return false;
	}

	//builder->setMaxBatchSize(_max_batch_size);
	//config->setMaxWorkspaceSize(64_MiB);

	//// build の設定
	// これ、どうなんやろうか。multi thread algorithm の方が良いんかな？
    //auto set_max_threads_status = builder->setMaxThreads(16);
	//if (!set_max_threads_status) {
	//	std::cout << "info string Error!: failed to setMaxThreads()" << std::endl;
	//	return false;
	//}

	// 4090 には関係ない
	auto DLA_cores = config->getDLACore();
	std::cout << "info string DLA_cores = " << DLA_cores << std::endl;

	// https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder_config.html#a0a88a9b43bbe47c839ba65de9b40779f
	// https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#a125336eeaa69c11d9aca0535449f0391
	//config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 256_MiB);
	//config->setBuilderOptimizationLevel(5);

	// TODO
	//     : builder->setMaxBatchSize(max_batch_size);
	//       -> https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder.html#a61518b4d3baceac8862b4ec3ae5c840e
	//          explicit batch dimension mode でcreat したnetwork にはno effect
	//     : fp16, int8
	// https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html
	//     : BuilderFlag の設定は、TensorRT が最速の物を探索するときに、より低精度な実装を選ぶようにTensorRT に示せる。
	if (_is_fp16 && builder->platformHasFastFp16()) {
		std::cout << "info string build fp16 engine" << std::endl;
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}
	else {
		std::cout << "info string build fp32 engine" << std::endl;
	}


	// https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_dims32.html#a7fbc72600de48c63141f3a83d9df0150
	//assert(network->getNbInputs() == 1);    // input の引数の数は1個

	auto profile = builder->createOptimizationProfile();

	const int32_t* ishape0 = network->getInput(0)->getDimensions().d;    // input_shape
	profile->setDimensions(
		network->getInput(0)->getName(),
		OptProfileSelector::kMIN,
		Dims4{1              , ishape0[1], ishape0[2], ishape0[3]}
	);
	profile->setDimensions(
		network->getInput(0)->getName(),
		OptProfileSelector::kOPT,
		Dims4{_max_batch_size, ishape0[1], ishape0[2], ishape0[3] }
	);
	profile->setDimensions(
		network->getInput(0)->getName(),
		OptProfileSelector::kMAX,
		Dims4{ _max_batch_size, ishape0[1], ishape0[2], ishape0[3] }
	);

	std::cout << "info string ishape0 = {" << ishape0[0] << ", " << ishape0[1] << ", " << ishape0[2] << ", " << ishape0[3] << "}" << std::endl;

#if defined(FEATURE_V2)    // 流石に3入力にはならんやろ...(フラグ)
	const int32_t* ishape1 = network->getInput(1)->getDimensions().d;    // input_shape
	profile->setDimensions(
		network->getInput(1)->getName(),
		OptProfileSelector::kMIN,
		Dims4{ 1              , ishape1[1], ishape1[2], ishape1[3] }
	);
	profile->setDimensions(
		network->getInput(1)->getName(),
		OptProfileSelector::kOPT,
		Dims4{ _max_batch_size, ishape1[1], ishape1[2], ishape1[3] }
	);
	profile->setDimensions(
		network->getInput(1)->getName(),
		OptProfileSelector::kMAX,
		Dims4{ _max_batch_size, ishape1[1], ishape1[2], ishape1[3] }
	);
	std::cout << "info string ishape1 = {" << ishape1[0] << ", " << ishape1[1] << ", " << ishape1[2] << ", " << ishape1[3] << "}" << std::endl;
	std::cout << "info string network->getInput(0)->getName()=" << network->getInput(0)->getName() << std::endl;
	std::cout << "info string network->getInput(1)->getName()=" << network->getInput(1)->getName() << std::endl;
	std::cout << "info string N_FEATURE_CHANNEL = " << N_FEATURE_CHANNEL << std::endl;
	std::cout << "info string N_FEATURE1_CHANNEL = " << N_FEATURE1_CHANNEL << std::endl;
	std::cout << "info string N_FEATURE2_CHANNEL = " << N_FEATURE2_CHANNEL << std::endl;
	std::cout << "info string N_FEATURE1_WHC = " << N_FEATURE1_WHC << std::endl;
	std::cout << "info string N_FEATURE2_WHC = " << N_FEATURE2_WHC << std::endl;
#endif

	int profileIdx = config->addOptimizationProfile(profile);    // プロファイルの登録
	if (profileIdx == -1) {
		return false;
	}

	//// build serialized engine
	auto serialized_engine = InferUniquePtr<nvinfer1::IHostMemory>(
		builder->buildSerializedNetwork(*network, *config)
	);
	if (!serialized_engine)    // 失敗
	{
		std::cout << "info string Error: filed to buildSerializedNetwork()" << std::endl;
		return false;
	}

	//// deserialize engine
	auto runtime = InferUniquePtr<nvinfer1::IRuntime>(
		nvinfer1::createInferRuntime(logger)
    );
	_engine.reset(
		runtime->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size())
    );
	if (!_engine) {
		std::cout << "info string Error: filed to deserializeCudaEngine()" << std::endl;
		return false;
	}

	return true;
}

bool NNTensorRT::load_model(const std::string& onnx_file_name) {
	std::string serialized_file_name;
	if (_is_fp16) {
		serialized_file_name
			= std::string(onnx_file_name)
			+ "." + std::to_string(_max_batch_size)
			+ ".fp16"
    		+ ".serialized";
	}
	else {
		serialized_file_name
			= std::string(onnx_file_name)
			+ "." + std::to_string(_max_batch_size)
			+ ".fp32"
			+ ".serialized";
	}
	std::ifstream serialized_file(serialized_file_name, std::ios::binary);

	//// dlshogi より
	if (serialized_file.is_open())    // 既にシリアル化したエンジンがあって、ファイルに保存されてる。
	{
		//// read
		serialized_file.seekg(0, std::ios_base::end);
		const size_t modelSize = serialized_file.tellg();

		serialized_file.seekg(0, std::ios_base::beg);
		std::unique_ptr<char[]> blob(new char[modelSize]);
		serialized_file.read(blob.get(), modelSize);

		//// deserialize
		auto runtime = InferUniquePtr<IRuntime>(createInferRuntime(logger));
		_engine = InferUniquePtr<nvinfer1::ICudaEngine>(
			runtime->deserializeCudaEngine(blob.get(), modelSize)
	    );    // シリアル化されている、先ほど読み込んだエンジンをデシリアライズ
	}
	else {
		if (!build_engine(onnx_file_name)) {
			std::cout << "info string Error!: failed to build_engine." << std::endl;
			return false;
		}

		// デシリアライズしたエンジンから、再度シリアル化されたエンジンのデータを取得
		auto serialized_engine = InferUniquePtr<nvinfer1::IHostMemory>(_engine->serialize());
		if (!serialized_engine) {
			std::cout << "info string Error!: failed to serialize()" << std::endl;
			return false;
		}

		// バイト列として、再度シリアル化したエンジンを書き出す
		std::ofstream serialized_file(serialized_file_name, std::ios::binary);
		if (!serialized_file) {
			std::cout << "info string Error!: failed to write serialized_file" << std::endl;
			return false;
		}
		serialized_file.write(
			static_cast<char*>(serialized_engine->data()),
			serialized_engine->size()
	    );
		if (serialized_file.fail()) {
			std::cout << "info string Error!: failed to write serialized_file" << std::endl;
			return false;
		}

		std::cout << "info: serialized_file_name = " << serialized_file_name << std::endl;
	}

	// contextの生成
	_context = InferUniquePtr<nvinfer1::IExecutionContext>(_engine->createExecutionContext());
	if (!_context)
	{
		std::cout << "info string ERROR : failed to mEngine->createExecutionContext()" << std::endl;
		return false;
	}

	// NN の入力のshape を保持しておく。
#if defined(FEATURE_V2)
	std::get<0>(_input_dims) = _engine->getBindingDimensions(0);
	std::get<1>(_input_dims) = _engine->getBindingDimensions(1);
#else
	_input_dims = _engine->getBindingDimensions(0);
#endif
	return true;
}


bool NNTensorRT::forward(const int& batch_size, const feature_t input, float* output_p, float* output_v) {
	//// batch_size は可変。ここで毎度設定。

	// HACK: 以下は二回ifdef してるけど、1回に纏める。
#if defined (FEATURE_V2)
	(std::get<0>(_input_dims)).d[0] = batch_size;
	bool status_sbd0 = _context->setBindingDimensions(0, std::get<0>(_input_dims));
	if (!status_sbd0) {
		std::cout << "info string Error: failed to setBindingDimensions()" << std::endl;
		return false;
	}

	(std::get<1>(_input_dims)).d[0] = batch_size;
	bool status_sbd1 = _context->setBindingDimensions(1, std::get<1>(_input_dims));
	if (!status_sbd1) {
		std::cout << "info string Error: failed to setBindingDimensions()" << std::endl;
		return false;
	}
#else
	_input_dims.d[0] = batch_size;
	bool status_sbd = _context->setBindingDimensions(0, _input_dims);
	if (!status_sbd) {
		std::cout << "info string Error: failed to setBindingDimensions()" << std::endl;
		return false;
	}
#endif


#if defined (FEATURE_V2)
	cudaMemcpyAsync(
		std::get<0>(_input_device), std::get<0>(input), SIZEOF_FEATURE1 * batch_size,
		cudaMemcpyHostToDevice, cudaStreamPerThread
	);
	cudaMemcpyAsync(
		std::get<1>(_input_device), std::get<1>(input), SIZEOF_FEATURE2 * batch_size,
		cudaMemcpyHostToDevice, cudaStreamPerThread
	);
#else
	cudaMemcpyAsync(
		_input_device, input, sizeof(float) * N_FEATURE_WHC * batch_size,
		cudaMemcpyHostToDevice, cudaStreamPerThread
	);
#endif

#if defined(USE_PACKED_FEATURE)
	unpack_feature1(std::get<0>(_unpacked_input_device), std::get<0>(_input_device), batch_size, cudaStreamPerThread);
	unpack_feature2(std::get<1>(_unpacked_input_device), std::get<1>(_input_device), batch_size, cudaStreamPerThread);
#endif

	const bool status = _context->enqueueV2(_input_bindings.data(), cudaStreamPerThread, nullptr);
	if (!status) {
		std::cout << "info string Error: failed to enqueueV2()";
		return false;
	}

	cudaMemcpyAsync(
		output_p, _output_p_device, sizeof(float) * N_LABEL_SIZE * batch_size,
		cudaMemcpyDeviceToHost, cudaStreamPerThread
    );
	cudaMemcpyAsync(
		output_v, _output_v_device, sizeof(float) * 1 * batch_size,
		cudaMemcpyDeviceToHost, cudaStreamPerThread
	);
	cudaStreamSynchronize(cudaStreamPerThread);    // host threadは、device のstream が終了するまでブロックされる。
	return true;
}