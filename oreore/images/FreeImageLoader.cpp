#include	"FreeImageLoader.h"


#if defined( FREEIMAGE_SUPPORT )


namespace OreOreLib
{

	FreeImageLoader::FreeImageLoader()
	{
		FreeImage_Initialise();
	}



	FreeImageLoader::~FreeImageLoader()
	{
		FreeImage_DeInitialise();
	}



	FIBITMAP* FreeImageLoader::Load( const tstring& filepath, bool bFloat, bool bFlip )
	{
		// Check(or estimate) filetype
#if defined(UNICODE) || defined(_UNICODE)
		auto fif = FreeImage_GetFileTypeU( filepath.c_str() );
#else
		auto fif = FreeImage_GetFileType( filepath.c_str() );
#endif

		if( fif == FIF_UNKNOWN )
#if defined(UNICODE) || defined(_UNICODE)
			FreeImage_GetFIFFromFilenameU( filepath.c_str() );
#else
			FreeImage_GetFIFFromFilename( filepath.c_str() );
#endif

		if( fif==FIF_UNKNOWN )
			return nullptr;

		// Load Image data
#if defined(UNICODE) || defined(_UNICODE)
		auto dib = FreeImage_LoadU( fif, filepath.c_str() );
#else
		auto dib = FreeImage_Load( fif, filepath.c_str() );
#endif
		if( !dib )
			return nullptr;

		// Flip
		if( bFlip )
			FreeImage_FlipVertical( dib );

		// Convert
		if( bFloat )
			dib = FreeImage_ConvertToRGBF( dib );
		else
			dib = FreeImage_ConvertTo32Bits( dib );

		return dib;
	}



	bool FreeImageLoader::Save( const tstring& filepath, const void* pData, uint32 width, uint32 height, uint32 pixelByteSize, bool bFlip )
	{
		// Check filetype from extension
//#if defined(UNICODE) || defined(_UNICODE)
//		FREE_IMAGE_FORMAT fif	= FreeImage_GetFIFFromFilenameU( filepath.c_str() );
//#else
//		FREE_IMAGE_FORMAT fif	= FreeImage_GetFIFFromFilename( filepath.c_str() );
//#endif

		// フォーマット不明 or 未対応フォーマットの場合は処理を中止する
		//if( fif == FIF_UNKNOWN || FreeImage_FIFSupportsReading( fif ) == FALSE )
		//	return false;

		// TODO: Implement GetFreeImageType
		//FREE_IMAGE_TYPE fit	= GetFreeImageType( texDesc.DataFormat );
		//auto bpp = pixelByteSize * 8;
		//FIBITMAP* dib = FreeImage_AllocateT( fit, width, height, bpp );
		
		//memcpy( FreeImage_GetBits( dib ), pData, width * height * pixelByteSize );
		
		//if( bFlip )
		//	FreeImage_FlipVertical( dib );

//#if defined(UNICODE) || defined(_UNICODE)
		//FreeImage_SaveU( fif, dib, filepath );
//#else
		//FreeImage_Save( fif, dib, filepath );
//#endif

		return true;
	}


}// end of namespace OreOreLib


#endif