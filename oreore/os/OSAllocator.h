#ifndef OS_ALLOCATOR_H
#define	OS_ALLOCATOR_H




namespace OreOreLib
{

	namespace OSAllocator
	{
		enum Status
		{
			Free,
			Reserved,
			Commited,

			NumStatus,
		};

		extern size_t PageSize();
		extern size_t AllocationGranularity();

		extern void* GetBaseAddress( const void* ptr );
		extern void* GetAllocationBase( const void* ptr );

		extern inline void* ReserveUncommited( size_t size, bool writable=true, bool executable=false );
		extern inline void* Commit( void* ptr, size_t size, bool writable=true, bool executable=false );
		extern inline void* CommitAligned( void* ptr, size_t size, size_t alignment, bool writable=true, bool executable=false );
		extern inline void* ReserveAndCommit( size_t size, bool writable=true, bool executable=false );

		extern inline bool Decommit( void* ptr, size_t size );
		extern inline bool DecommitAligned( void* ptr, size_t size, size_t alignment );
		extern inline bool Release( void* ptr );

		extern inline void* ReallocateAndCommit( void* pSrc, size_t srcSize, size_t newSize, bool writable=true, bool executable=false );

		extern inline void* FindRegion( void* ptr, Status status, size_t regionSize, size_t searchRange );


		extern inline bool IsFullyReserved( void* ptr, size_t regionSize );
		extern inline bool IsFullyCommited( void* ptr, size_t regionSize );
		extern inline bool IsReleased( void* ptr );

		extern void DisplayMemoryInfo( const void* ptr );

		
		struct SystemInfo
		{
			size_t	m_PageSize;
			size_t	m_AllocationGranularity;

			SystemInfo();
		};


	}// end of namespace


}// end of namespace



#endif // !OS_ALLOCATOR_H
