#ifndef FREE_IMAGE_LOADER_H
#define	FREE_IMAGE_LOADER_H

#if defined( FREEIMAGE_SUPPORT )

#include	<FreeImage.h>
#pragma comment( lib, "freeimage.lib" )


#include	<oreore/common/Utility.h>


namespace OreOreLib
{

	class FreeImageLoader
	{
	public:

		FreeImageLoader();
		virtual ~FreeImageLoader();

		FIBITMAP* Load( const tstring& filepath, bool bFloat, bool bFlip );
		bool Save( const tstring& filepath, const void* pData, uint32 width, uint32 height, uint32 pixelByteSize, bool bFlip );
	};


}// end of namespace OreOreLib

#endif

#endif // !FREE_IMAGE_LOADER_H
