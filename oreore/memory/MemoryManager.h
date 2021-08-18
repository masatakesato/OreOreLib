#ifndef	MEMORY_MANAGER_H
#define	MEMORY_MANAGER_H



#include	"PoolAllocator.h"
//#include	"MemoryAllocator.h"
//#include	"TLSF.h"




namespace OreOreLib
{


	class MemoryManager
	{
	public:

		MemoryManager();// Default constructor
		//MemoryManager( int blockSize, int pageSize );// Constructor

		MemoryManager( const MemoryManager& obj );// Copy constructor
		MemoryManager( MemoryManager&& obj );// Move constructor

		~MemoryManager();

		MemoryManager& operator=( const MemoryManager& obj );// Copy assignment operator
		MemoryManager& operator=( MemoryManager&& obj );// Move assignment operator


		void* Allocate( size_t size, size_t alignment=ByteSize::DefaultAlignment );
		void* Callocate( size_t n, size_t size, size_t alignment=ByteSize::DefaultAlignment );
		void* Reallocate( void*& mem, size_t size, size_t alignment=ByteSize::DefaultAlignment );
		bool Free( void*& mem );
		
		void Cleanup();

		void Display() const;



	private:

		static const int32 c_NumPoolTables = 44;
		static const size_t c_BlockSizes[ c_NumPoolTables ];

		static const int32 c_NumAllocSizes = 8;
		static const size_t c_AllocSizes[ c_NumAllocSizes ];


		PoolAllocator	m_PoolTable[ c_NumPoolTables ];
		
		static const int32 c_NumSizes = 32768;
		PoolAllocator*	m_pSizeToPoolTable[ c_NumSizes ];


		void Init();


	};



}// end of namespace


#endif // !MEMORY_MANAGER_H
