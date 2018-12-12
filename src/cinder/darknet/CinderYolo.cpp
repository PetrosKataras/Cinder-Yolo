#include "CinderYolo.h"
#include "cinder/ip/Resize.h"
#include "cinder/Filter.h"
#include "cinder/Log.h"
#include "cinder/app/AppBase.h"

extern "C" float get_color( int c, int x, int max );
namespace cinder { namespace yolo {

CinderYolo::CinderYolo( const fs::path& cfgFilepath, const fs::path& weightsFilepath, const fs::path& labelsFilepath )
{
	// Load the network
	auto cfgFilepathStr = cfgFilepath.string();
	auto weightsFilepathStr = weightsFilepath.string();
	mDetector = std::unique_ptr<Detector>( new Detector( &cfgFilepathStr[0], &weightsFilepathStr[0] ) );
	// Load labels ( if defined )
	if( ! labelsFilepath.empty() ) { 
		auto labelsFilepathStr = labelsFilepath.string();
		std::ifstream namesFile( labelsFilepathStr );
		if( namesFile.is_open() ) {
			for( std::string line; getline( namesFile, line ); ) {
				mLabels.push_back( line );
			}
		}
	}
	// Create the input queue
	mSurfaceQueue = std::make_unique<ConcurrentCircularBuffer<Surface>>( 15 );
	// Start the processing thread
	auto futureObj = mTerminateProcessSignal.get_future();
	mNetworkProcessThread = std::thread( &CinderYolo::networkProcessFn, this, std::move( futureObj ) );
}

CinderYolo::~CinderYolo()
{
	// Exit and terminate the processing thread
	mTerminateProcessSignal.set_value();
	mNetworkProcessThread.join();
	// Clear the queues
	mSurfaceQueue->cancel();

	std::lock_guard<std::mutex> guard( mMutex );
	mDetections.clear();
	
}

void CinderYolo::runYolo( const Surface& surface, const float threshold )
{
	mSurfaceQueue->tryPushFront( surface );	
	mThreshold = threshold;
}

void CinderYolo::networkProcessFn( std::future<void> futureObj )
{
	while( futureObj.wait_for( std::chrono::milliseconds( 1 ) ) == std::future_status::timeout ) {
		if( mDetector && mSurfaceQueue->isNotEmpty() ) {
			Surface surface;
			Surface surfaceCopy;
			ci::vec2 scaleBRect( 1.0f );
			if( mSurfaceQueue->tryPopBack( &surface ) ) {
				if( surface.getWidth() != mDetector->get_net_width() || surface.getHeight() != mDetector->get_net_height() ) {
					surfaceCopy = ip::resizeCopy( surface, surface.getBounds(), ivec2( mDetector->get_net_width(), mDetector->get_net_height() ) );
					scaleBRect.x = (float)surface.getWidth() / (float)mDetector->get_net_width();
					scaleBRect.y = (float)surface.getHeight() / (float)mDetector->get_net_height();
				}
				image_t yoloImage = surfaceToDarknetImage( surfaceCopy );
				auto result = mDetector->detect( yoloImage, mThreshold );
				Detector::free_image( yoloImage );
				std::lock_guard<std::mutex> guard( mMutex );
				mDetections.clear();
				for( auto& d : result ) {
					Detection detection;
					detection.mBoundingRect = Rectf( d.x, d.y, d.x+d.w, d.y+d.h );
					detection.mBoundingRect.scale( scaleBRect );
					detection.mColor = getColorFromClassId( d.obj_id );
					detection.mLabel = getLabelFromClassId( d.obj_id );
					mDetections.push_back( detection );
				}
			}
		}
		else std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) ) ;
	}
}

image_t CinderYolo::surfaceToDarknetImage( const Surface& surface )
{
	// conversion routine from: darknet/src/image_opencv.cpp 	
	auto makeYoloImage = [] ( int w, int h, int c ) -> image_t {
		image_t out;
		out.data = (float*)calloc( h*w*c, sizeof( float ) );
		out.w = w;
		out.h = h;
		out.c = c; 
		return out;
	};
	int w = surface.getWidth();
	int h = surface.getHeight();
	int c = surface.getPixelInc();
	image_t yoloImage = makeYoloImage( w, h, c );
	int widthStep = w * c;
	for( int i = 0; i < h; ++i ) {
		for( int k = 0; k < c ; ++k ) {
			for( int j = 0; j < w; ++j ) {
				yoloImage.data[ k*w*h+i*w+j ] = (float)surface.getData()[ i*widthStep+j*c+k ] / 255.0f;
			}
		}
	}
	return yoloImage;
}

ci::Colorf CinderYolo::getColorFromClassId( const int classId )
{
	int numClasses = mDetector->get_num_classes();
	int offset = classId * 123457 % numClasses;
	float r = get_color( 2, offset, numClasses );
	float g = get_color( 1, offset, numClasses );
	float b = get_color( 0, offset, numClasses );
	return ci::Colorf( r, g, b );
}

std::string CinderYolo::getLabelFromClassId( const int classId )
{
	return mLabels.size() > 0 && mLabels.size() > classId ? mLabels[ classId ] : std::string();
}

} // namespace yolo
} // namespace cinder
