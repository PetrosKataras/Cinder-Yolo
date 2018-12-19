#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/Log.h"

// NDI
#include "CinderNDIReceiver.h"
#include "CinderNDIFinder.h"

// Yolo
#include "cinder/darknet/CinderYolo.h"

using namespace ci;
using namespace ci::app;

class NDIYoloApp : public App {
public:
    void setup() final;
    void update() final;
    void draw() final;
	void keyDown( KeyEvent event ) final;
	void cleanup() final;
private: 
	void sourceAdded( const NDISource& source );
	void sourceRemoved( const std::string sourceName );
private:
	CinderNDIFinderPtr mCinderNDIFinder;
	CinderNDIReceiverPtr mCinderNDIReceiver;
	ci::signals::Connection mNDISourceAdded;
	ci::signals::Connection mNDISourceRemoved;

	std::unique_ptr<yolo::CinderYolo> mCiDarknet;

	SurfaceRef mSurface;
	std::vector<yolo::CinderYolo::Detection> mDetectedObjects;
	gl::TextureRef mTexture;
	float mThreshold{ .1f };
};

void NDIYoloApp::cleanup()
{
	if( mCinderNDIFinder ) {
		mNDISourceAdded.disconnect();	
		mNDISourceRemoved.disconnect();
	}
}

void NDIYoloApp::sourceAdded( const NDISource& source )
{
	std::cout << "NDI source added: " << source.p_ndi_name <<std::endl;
	// Create the NDI receiver for this source
	if( ! mCinderNDIReceiver ) {
		CinderNDIReceiver::Description recvDscr;
		recvDscr.source = &source;
		mCinderNDIReceiver = std::make_unique<CinderNDIReceiver>( recvDscr );
	}
	else
		mCinderNDIReceiver->connect( source );
}

void NDIYoloApp::sourceRemoved( std::string sourceName )
{
	std::cout << "NDI source removed: " << sourceName <<std::endl;
}

void NDIYoloApp::setup()
{
	// Create Yolo detector
	mCiDarknet = std::make_unique<yolo::CinderYolo>( getAssetPath( "yolov3-tiny.cfg" ), getAssetPath( "yolov3-tiny.weights" ), getAssetPath( "coco.names" ) );

	// Create the NDI finder
	CinderNDIFinder::Description finderDscr;
	mCinderNDIFinder = std::make_unique<CinderNDIFinder>( finderDscr );
	
	mNDISourceAdded = mCinderNDIFinder->getSignalNDISourceAdded().connect(
		std::bind( &NDIYoloApp::sourceAdded, this, std::placeholders::_1 )
	);
	mNDISourceRemoved = mCinderNDIFinder->getSignalNDISourceRemoved().connect(
		std::bind( &NDIYoloApp::sourceRemoved, this, std::placeholders::_1 )
	);
}

void NDIYoloApp::update()
{
	//getWindow()->setTitle( std::to_string( getAverageFps() ) );
	if( mCinderNDIReceiver ) {
		mTexture = mCinderNDIReceiver->getVideoTexture();
		if( mTexture )
			mSurface = ci::Surface::create( mTexture->createSource() );
	}

	if( mCiDarknet && mSurface ) {
		mCiDarknet->runYolo( *mSurface.get(), mThreshold );
	}
}

void NDIYoloApp::draw()
{
	gl::clear( Color( .2f, .2f, .2f ) );
	auto detections = mCiDarknet->getDetections();
	gl::draw( mTexture );
	for( const auto& detectedObject : detections ) {
			gl::ScopedColor scopedColor( detectedObject.mColor );
			gl::drawStrokedRect( detectedObject.mBoundingRect );
			gl::ScopedBlend scopedBlend( true );
			auto labelBgUL = detectedObject.mBoundingRect.getUpperLeft() - vec2( .0f, 20.f );
			auto labelBgBR = labelBgUL + vec2( 70.0f, 20.f );
			gl::drawSolidRect( Rectf( labelBgUL, labelBgBR ) ); 
			gl::drawString( detectedObject.mLabel, detectedObject.mBoundingRect.getUpperLeft() - vec2( 0.f, 10.f ), Color::black() );
	}
}

void NDIYoloApp::keyDown( KeyEvent event )
{
	if( event.getChar() == 'i' ) {
		mThreshold += .05f;
		mThreshold = std::min( mThreshold, 1.0f );
		std::cout << "THRES " << mThreshold << std::endl;
	}
	else if( event.getChar() == 'd' ) {
		mThreshold -= .05f;
		mThreshold = std::max( mThreshold, 0.0f );
		std::cout << "THRES " << mThreshold << std::endl;
	}
}

CINDER_APP( NDIYoloApp, RendererGl );
