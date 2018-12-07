#include "yolo_v2_class.hpp"
#include "cinder/ConcurrentCircularBuffer.h"
#include "cinder/Surface.h"
#include <thread>
#include <future>
#include <chrono>
#include <atomic>

namespace cinder { namespace yolo {

class CinderYolo {
public:
	struct Detection {
		Rectf mBoundingRect;
		ci::Colorf mColor;
		std::string mLabel;
		float mProbability{ 0.f };
	};
	using Detections = std::vector<Detection>;

	CinderYolo( const fs::path& cfgFilepath, const fs::path& weightsFilepath, const fs::path& labelsFilepath = fs::path() );
	~CinderYolo();
	void runYolo( const Surface32f& pixels, const float threshold );
	const Detections getDetections() const { return mDetections; }
private:
	void networkProcessFn(std::future<void> test);
	image_t surfaceToDarknetImage( const Surface32f& surface );
	ci::Colorf getColorFromClassId( const int classId );
	std::string getLabelFromClassId( const int classId );
private:
	std::unique_ptr<Detector> mDetector;
	std::thread mNetworkProcessThread;
	std::promise<void> mTerminateProcessSignal;
	std::unique_ptr<ConcurrentCircularBuffer<Surface32f>> mSurfaceQueue;
	Detections mDetections;
	std::atomic<float> mThreshold{ 0.4f };
	std::vector<std::string> mLabels;
	std::mutex mMutex;
};


} // namespace yolo
} // namespace cinder
