#ifndef FREE_IMAGE_LOADER_H
#define	FREE_IMAGE_LOADER_H

#if defined( FREEIMAGE_SUPPORT )

#include	<FreeImage.h>
#pragma comment( lib, "freeimage.lib" )

#include	"../common/Utility.h"
#include	"DataFormat.h"



namespace OreOreLib
{
	//##############################################################################//
	//																				//
	//								FreeImageLoader class							//
	//																				//
	//##############################################################################//

	class FreeImageLoader
	{
	public:

		FreeImageLoader();
		virtual ~FreeImageLoader();

		FIBITMAP* Load( const tstring& filepath, bool bFloat, bool bFlip );
		bool Save( const tstring& filepath, const void* pData, uint32 width, uint32 height, uint32 pixelByteSize, bool bFlip );
	};




	//##############################################################################//
	//																				//
	//								Helper functions								//
	//																				//
	//##############################################################################//

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



	static FREE_IMAGE_TYPE GetFreeImageType( DATA_FORMAT data_format )
	{
		switch( data_format )
		{
		case FORMAT_UNKNOWN:// unknown type
			return FIT_UNKNOWN;

		case FORMAT_R8G8B8_UNORM:// standard image			: 1-, 4-, 8-, 16-, 24-, 32-bit
			return FIT_BITMAP;

		case FORMAT_B8G8R8_UNORM:
			return FIT_BITMAP;

		case FORMAT_R16_UNORM:// array of unsigned short	: unsigned 16-bit
			return FIT_UINT16;

		case FORMAT_R16_SINT:// array of short			: signed 16-bit
			return FIT_INT16;

		case FORMAT_R32_UINT:// array of unsigned long	: unsigned 32-bit
			return FIT_UINT32;

		case FORMAT_R32_SINT:// array of long			: signed 32-bit
			return FIT_INT32;

		case FORMAT_R32_FLOAT:// array of float			: 32-bit IEEE floating point
			return FIT_FLOAT;

			//case FORMAT_UNKNOWN:// array of double			: 64-bit IEEE floating point
			//	return FIT_DOUBLE;

			//case FORMAT_UNKNOWN:// array of FICOMPLEX		: 2 x 64-bit IEEE floating point
			//	return FIT_COMPLEX;

		case FORMAT_R16G16B16_UNORM:// 48-bit RGB image			: 3 x 16-bit
			return FIT_RGB16;

		case FORMAT_R16G16B16_FLOAT:// 48-bit RGB float image			: 3 x 16-bit
			return FIT_RGB16;

		case FORMAT_R16G16B16A16_UNORM:// 64-bit RGBA image		: 4 x 16-bit
			return FIT_RGBA16;

		case FORMAT_R16G16B16A16_FLOAT:// 64-bit RGBA float image		: 4 x 16-bit
			return FIT_RGBA16;

		case FORMAT_R32G32B32_FLOAT:// 96-bit RGB float image	: 3 x 32-bit IEEE floating point
			return FIT_RGBF;

		case FORMAT_R32G32B32A32_FLOAT:// 128-bit RGBA float image	: 4 x 32-bit IEEE floating point
			return FIT_RGBAF;

		default:// unknown type
			return FIT_UNKNOWN;

		};

	}



	static DATA_FORMAT GetDataFormat( FIBITMAP* dib, bool bSRGB=false )
	{
		auto fit = FreeImage_GetImageType( dib );
		auto bpp = FreeImage_GetBPP( dib );

		switch( fit )
		{
		case FIT_UNKNOWN:// unknown type
			return FORMAT_UNKNOWN;

		case FIT_BITMAP:// standard image			: 1-, 4-, 8-, 16-, 24-, 32-bit
			if( bpp == 8 )		return FORMAT_R8_UNORM;
			//else if(bpp==16)	return FORMAT_G8R8_UNORM;
			else if( bpp==24 )	return bSRGB ? FORMAT_B8G8R8_SRGB : FORMAT_B8G8R8_UNORM;
			else if( bpp==32 )	return bSRGB ? FORMAT_B8G8R8A8_UNORM_SRGB : FORMAT_B8G8R8A8_UNORM;

			return bSRGB ? FORMAT_B8G8R8A8_UNORM_SRGB : FORMAT_B8G8R8A8_UNORM;//return FORMAT_R8G8B8A8_UNORM;

		case FIT_UINT16:// array of unsigned short	: unsigned 16-bit
			return FORMAT_R16_UNORM;

		case FIT_INT16:// array of short			: signed 16-bit
			return FORMAT_R16_SINT;

		case FIT_UINT32:// array of unsigned long	: unsigned 32-bit
			return FORMAT_R32_UINT;

		case FIT_INT32:// array of long			: signed 32-bit
			return FORMAT_R32_SINT;

		case FIT_FLOAT:// array of float			: 32-bit IEEE floating point
			return FORMAT_R32_FLOAT;

		case FIT_DOUBLE:// array of double			: 64-bit IEEE floating point
			return FORMAT_UNKNOWN;

		case FIT_COMPLEX:// array of FICOMPLEX		: 2 x 64-bit IEEE floating point
			return FORMAT_UNKNOWN;

		case FIT_RGB16:// 48-bit RGB image			: 3 x 16-bit
			return FORMAT_R16G16B16_UNORM;

		case FIT_RGBA16:// 64-bit RGBA image		: 4 x 16-bit
			return FORMAT_R16G16B16A16_UNORM;

		case FIT_RGBF:// 96-bit RGB float image	: 3 x 32-bit IEEE floating point
			return FORMAT_R32G32B32_FLOAT;

		case FIT_RGBAF:// 128-bit RGBA float image	: 4 x 32-bit IEEE floating point
			return FORMAT_R32G32B32A32_FLOAT;

		default:// unknown type
			return FORMAT_UNKNOWN;

		};
	}



}// end of namespace OreOreLib

#endif

#endif // !FREE_IMAGE_LOADER_H
