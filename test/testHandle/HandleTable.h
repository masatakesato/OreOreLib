#ifndef HANDLE_TABLE_H
#define	HANDLE_TABLE_H



// http://ce.eng.usc.ac.ir/files/1511334027376.pdf
// p.252 defragmentation

// http://smallmemory.com/6_AllocationChapter.pdf
// p.35 Handle class implementation


// Handle table?


#include	<oreore/MathLib.h>
#include	<oreore/container/RingQueue.h>
#include	<oreore/common/Utility.h>



namespace OreOreLib
{


	template< typename T >
	using Handle = T**;

	typedef uint64  Handle_t;


	template< class T >
	class HandleTable
	{
	public:

		HandleTable()
		{

		}

		HandleTable( int numhandles )
		{
			m_FreeHandles.Init( numhandles );
			m_Reserved.Init( numhandles );
			m_pData.Init( numhandles );
		}

		HandleTable( const HandleTable& obj )
			: m_FreeHandles( obj.m_FreeHandles )
			, m_Reserved( obj.m_Reserved )
			, m_pData( obj.m_pData )
		{

		}

		~HandleTable()
		{
			// Destruct free/occupied handle list
			m_FreeHandles.Release();
			m_Reserved.Release();

			// Release memory data
			m_pData.Release();
		}


		// 
		Handle_t AddHandle()
		{

			m_FreeHandles.Extend(1);
			m_Reserved.Extend(1);

			m_pData.Extend(1);

			return 0;
		}

		void RemoveHandle( Handle_t h )
		{
			

		}



		inline HandleTable& operator=( const HandleTable& obj )
		{
			if( this == obj )
				return *this;

			m_FreeHandles	= obj.m_FreeHandles;
			m_Reserved		= obj.m_Reserved;
			m_pData			= obj.m_pData;

			return *this;
		}


		void SetHandle( Handle_t h, T *pData )
		{
			assert( h < m_pData.Length() );

			*( m_pData + h ) = pData;
		}


		T* GetHandle( Handle_t h ) const
		{
			assert( h < m_pData.Length() );
			return m_pData[h];
		}



	private:

		// free(occupied) handle list
		RingQueue<uint32>	m_FreeHandles;
		Memory<bool>	m_Reserved;

		// Actual Memory data
		Memory< void* >	m_pData;

	};


}// end of namespace


#endif // !HANDLE_TABLE_H
