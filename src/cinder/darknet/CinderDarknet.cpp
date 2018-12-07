#include "CinderDarknet.h"
#include "image.h"
#include "cinder/ip/Resize.h"
#include "cinder/Filter.h"
#include "cinder/Log.h"
#include "cinder/app/AppBase.h"

#if defined( GPU )
	#define BLOCK 512

	#include "cuda_runtime.h"
	#include "curand.h"
	#include "cublas_v2.h"

	#if defined( CUDNN )
		#include "cudnn.h"
	#endif
#endif

namespace cinder { namespace darknet {

CinderDarknet::CinderDarknet( const fs::path& cfgFilepath, const fs::path& weightsFilepath, const fs::path& labelsFilepath )
{
	// Load the network
	auto cfgFilepathStr = cfgFilepath.string();
	auto weightsFilepathStr = weightsFilepath.string();
	cuda_set_device( 0 );
	mNet = std::unique_ptr<network, DarknetDeleter>( load_network( &cfgFilepathStr[0], &weightsFilepathStr[0], 0 ) );
	set_batch_network( mNet.get(), 1 );
	// Load labels ( if defined )
	if( ! labelsFilepath.empty() ) { 
		auto labelsFilepathStr = labelsFilepath.string();
		mLabels = get_labels( &labelsFilepathStr[0] );
	}
	// Create the input queue
	mSurfaceQueue = std::make_unique<ConcurrentCircularBuffer<Surface32f>>( 15 );
	mDetectionsQueue = std::make_unique<ConcurrentCircularBuffer<Detection>>( 15 );
	// Start the processing thread
	auto futureObj = mTerminateProcessSignal.get_future();
	mNetworkProcessThread = std::thread( &CinderDarknet::networkProcessFn, this, std::move( futureObj ) );
}

CinderDarknet::~CinderDarknet()
{
	// Exit and terminate the processing thread
	mTerminateProcessSignal.set_value();
	mNetworkProcessThread.join();
	// Clear the queues
	mSurfaceQueue->cancel();
	mDetectionsQueue->cancel();
	
}

void CinderDarknet::runYolo( const Surface32f& surface, const float threshold )
{
	mSurfaceQueue->tryPushFront( surface );	
	mThreshold = threshold;
}

void CinderDarknet::networkProcessFn( std::future<void> futureObj )
{
	while( futureObj.wait_for( std::chrono::milliseconds( 1 ) ) == std::future_status::timeout ) {
		if( mNet && mSurfaceQueue->isNotEmpty() ) {
			Surface32f surface;
			Surface32f surfaceCopy;
			if( mSurfaceQueue->tryPopBack( &surface ) ) {
				if( surface.getWidth() != mNet->w || surface.getHeight() != mNet->h ) {
					surfaceCopy = ip::resizeCopy( surface, surface.getBounds(), ivec2( mNet->w, mNet->h ) );
				}
				image darknetImage = surfaceToDarknetImage( surfaceCopy );
				network_predict( mNet.get(), darknetImage.data );
				free_image( darknetImage );
				int numBoxes{ 0 };
				auto* detections = get_network_boxes( mNet.get(), darknetImage.w, darknetImage.h, mThreshold, 0.0f, 0, 1, &numBoxes );
				auto layer = mNet->layers[ mNet->n - 1 ];
				do_nms_sort( detections, numBoxes, layer.classes, 0.4f );
				for( size_t i = 0; i < numBoxes; ++i ) {
					auto box = detections[i].bbox;
					const int objId = max_index( detections[i].prob, layer.classes );
					const float prob = detections[i].prob[ objId ];

					if( prob > mThreshold && objId >= 0 ) {
						Detection detection;
						int left  = ( box.x - box.w / 2.f ) * mNet.get()->w;
						int right = ( box.x + box.w / 2.f ) * mNet.get()->w;
						int top   = ( box.y - box.h / 2.f ) * mNet.get()->h;
						int bot   = ( box.y + box.h / 2.f ) * mNet.get()->h;

						if( left < 0 ) left = 0;
						if( right > mNet.get()->w-1 ) right = mNet.get()->w-1;
						if( top < 0 ) top = 0;
						if( bot > mNet.get()->h-1 ) bot = mNet.get()->h-1;
						detection.mBoundingRect = Rectf( vec2( left, top ), vec2( right, bot ) );
						auto scaleBoundingRect = ci::vec2( (float)surface.getWidth() / (float)mNet->w, (float)surface.getHeight() / (float)mNet->h );
						detection.mBoundingRect.scale( scaleBoundingRect );

						int offset = objId * 123457 % layer.classes;
						float r = get_color( 2, offset, layer.classes );
						float g = get_color( 1, offset, layer.classes );
						float b = get_color( 0, offset, layer.classes );
						detection.mColor = Colorf( r, g, b );
						if( mLabels ) {
							detection.mLabel = mLabels[ objId ];
						}
						detection.mProbability = prob;
						CI_LOG_I( " Bounding rect : " << detection.mBoundingRect << " RGB " << detection.mColor << " LABEL : " << detection.mLabel << " PROB : " << detection.mProbability );
						mDetectionsQueue->tryPushFront( detection );
					}
				}
				free_detections( detections, numBoxes );
			}
		}
	}
}

image CinderDarknet::surfaceToDarknetImage( const Surface32f& surface )
{
	// conversion routine from: darknet/src/image_opencv.cpp 	
	int w = surface.getWidth();
	int h = surface.getHeight();
	int c = surface.getPixelInc();
	image darknetImage = make_image( w, h, c );
	int widthStep = w * c;
	for( int i = 0; i < h; ++i ) {
		for( int k = 0; k < c ; ++k ) {
			for( int j = 0; j < w; ++j ) {
				darknetImage.data[ k*w*h + i*w + j ] = surface.getData()[ i*widthStep + j*c + k ];
			}
		}
	}
	return darknetImage;
}

} // namespace darknet
} // namespace cinder
