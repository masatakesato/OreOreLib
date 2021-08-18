#include	"BoundaryTagBlock.h"

#include	<assert.h>

#include	"../common/TString.h"



namespace OreOreLib
{
	//default constructor
	BoundaryTagBlock::BoundaryTagBlock()
		: m_IsFree( true )
		, m_DataSize( 0 )
		, m_pData( nullptr )
		, m_pEnd( nullptr )
		, next( this )
		, prev( this )
	{

	}


	// constructor
	BoundaryTagBlock::BoundaryTagBlock( uint8* data, uint32 size, bool isfree )
		: m_IsFree( isfree )
		, m_DataSize( size )
		, m_pData( data )
		, m_pEnd( (uint32 *)( data + size ) )// set end tag address.
		, next( this )
		, prev( this )
	{
		assert( data != nullptr );
		*m_pEnd	= c_BoundaryTagAllocSize + size;// set reserved memory amount ( boundary tag obj + datasize + end tag size)
	}


	//destructor
	BoundaryTagBlock::~BoundaryTagBlock()
	{

	}


	void BoundaryTagBlock::Info()
	{
		tcout << _T( "BoundaryTagBlock info...\n" );
		tcout << _T( "    usage: " ) << ( m_IsFree ? _T( "free" ) : _T( "reserved" ) ) << tendl;
		tcout << _T( "    alloc size: " ) << *m_pEnd << _T( " [bytes]. \n" );
		tcout << _T( "    data size: " ) << m_DataSize << _T( " [bytes]. \n" );
	}



}// end of namespace