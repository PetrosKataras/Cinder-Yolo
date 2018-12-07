#include "darknet.h"
#include "cinder/Surface.h"
#include <thread>
#include <future>
#include <chrono>
#include <atomic>
#include "cinder/ConcurrentCircularBuffer.h"

namespace cinder { namespace darknet {

class CinderDarknet {
public:
	struct Detection {
		Rectf mBoundingRect;
		ci::Colorf mColor;
		std::string mLabel;
		float mProbability{ 0.f };
	};
	CinderDarknet( const fs::path& cfgFilepath, const fs::path& weightsFilepath, const fs::path& labelsFilepath = fs::path() );
	~CinderDarknet();
	void runYolo( const Surface32f& pixels, const float threshold );
	ConcurrentCircularBuffer<Detection>* getDetectionsQueue() const { return mDetectionsQueue.get(); }
private:
	void networkProcessFn(std::future<void> test);
	struct DarknetDeleter {
		void operator()( network* net ) { free_network( net ); }
	};
	image surfaceToDarknetImage( const Surface32f& surface );
private:
	std::unique_ptr<network, DarknetDeleter> mNet;
	std::thread mNetworkProcessThread;
	std::promise<void> mTerminateProcessSignal;
	std::unique_ptr<ConcurrentCircularBuffer<Surface32f>> mSurfaceQueue;
	std::unique_ptr<ConcurrentCircularBuffer<Detection>> mDetectionsQueue;
	std::atomic<float> mThreshold{ 0.4f };
	char** mLabels = nullptr;
};


} // namespace darknet
} // namespace cinder
