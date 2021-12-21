#ifndef IMAGE_LOADER_H
#define	IMAGE_LOADER_H

#include	<oreore/common/Utility.h>



namespace OreOreLib
{

	class ImageLoader
	{
	public:

		ImageLoader();
		~ImageLoader();

		bool Load( const tstring& filepath, bool bFloat, bool bFlip );
		bool Save( const tstring& filepath, const void* pData, uint32 width, uint32 height, uint32 pixelByteSize, bool bFlip );
	};


}// end of namespace OreOreLib


#endif // !IMAGE_LOADER_H
