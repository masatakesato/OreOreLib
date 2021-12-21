#include	"ImageLoader.h"

#if defined( FREEIMAGE_SUPPORT )
#include	<FreeImage.h>
#pragma comment( lib, "freeimage.lib" )
#endif



namespace OreOreLib
{

	static uint32 FINumChannels( FIBITMAP* dib )
	{
		auto coltype = FreeImage_GetColorType( dib );

		switch( coltype )
		{
		case FIC_MINISWHITE:
			return 1;

		case FIC_MINISBLACK:
			return 1;

		case FIC_RGB:
			return 3;

		case FIC_PALETTE:
			return 3;

		case FIC_RGBALPHA:
			return 4;

		case FIC_CMYK:
			return 4;

		default:
			return -1;
		}

	}




	ImageLoader::ImageLoader()
	{
		FreeImage_Initialise();
	}



	ImageLoader::~ImageLoader()
	{
		FreeImage_DeInitialise();
	}



	bool ImageLoader::Load( const tstring& filepath, bool bFloat, bool bFlip )
	{
		// Check(or estimate) filetype
		auto fif = FreeImage_GetFileTypeU( filepath.c_str() );

		if( fif == FIF_UNKNOWN )
			/*FreeImage_GetFIFFromFilename*/FreeImage_GetFIFFromFilenameU( /*filename*/filepath.c_str() );

		if( fif==FIF_UNKNOWN )
			return false;

		// Load Image data
		auto dib = FreeImage_LoadU( fif, filepath.c_str() );
		if( !dib )
			return false;

		// Flip
		if( bFlip )
			FreeImage_FlipVertical( dib );

		// Convert
		if( bFloat )
			dib = FreeImage_ConvertToRGBF( dib );
		else
			dib = FreeImage_ConvertTo32Bits( dib );


		auto width	= FreeImage_GetWidth( dib );
		auto height	= FreeImage_GetHeight( dib );
		auto depth	= FINumChannels( dib );

		// Create tensor from raw pixel array data
		auto rawdata = FreeImage_GetBits( dib );





		// Uload Freeimage buffer
		FreeImage_Unload( dib );

		return true;
	}



	bool ImageLoader::Save( const tstring& filepath, const void* pData, uint32 width, uint32 height, uint32 pixelByteSize, bool bFlip )
	{
		//// Check filetype from extension
		//FREE_IMAGE_FORMAT fif	= /*FreeImage_GetFIFFromFilename*/FreeImage_GetFIFFromFilenameU( filepath.c_str() );

		//// フォーマット不明 or 未対応フォーマットの場合は処理を中止する
		//if( fif == FIF_UNKNOWN || FreeImage_FIFSupportsReading( fif ) == FALSE )
		//	return false;

// TODO: Implement GetFreeImageType
		//FREE_IMAGE_TYPE fit	= GetFreeImageType( texDesc.DataFormat );
		//auto bpp = pixelByteSize * 8;
		//FIBITMAP* dib = FreeImage_AllocateT( fit, width, height, bpp );
		
		//memcpy( FreeImage_GetBits( dib ), pData, width * height * pixelByteSize );
		
		//if( bFlip )
		//	FreeImage_FlipVertical( dib );

		//FreeImage_SaveU( fif, dib, filepath );

		return true;
	}


}// end of namespace OreOreLib