#ifndef ARRAY_VIEW_H
#define	ARRAY_VIEW_H

#include	"../common/TString.h"
//#include	"../memory/Memory.h"

#include	"ArrayBase.h"



// https://www.codeproject.com/Articles/848746/ArrayView-StringView



namespace OreOreLib
{


	template< typename T, typename IndexType >
	class ArrayBase< detail::ARRVIEW<T>, detail::DynamicSize, IndexType > : public MemoryBase<T, IndexType>
	{
		//using IndexType = typename MemoryBase<T, IndexType>::IndexType;
		using Ptr = T*;
		using ConstPtr = const T*;

	public:

		// Default constructor
		ArrayBase()
			: MemoryBase<T, IndexType>()
		{
		}


		// Constructor
		ArrayBase( ConstPtr const pdata, IndexType length )
		{
			Init( pdata, length );
		}


		// Constructor
		ArrayBase( const MemoryBase<T, IndexType>& obj )
		{
			Init( obj );
		}


		// Destructor
		~ArrayBase()
		{
			Release();
		}


		// Copy constructor
		ArrayBase( const ArrayBase& obj )
		{
			this->m_pData		= obj.m_pData;
			this->m_Length		= obj.m_Length;
			this->m_AllocSize	= 0;
			this->m_Capacity		= obj.m_Length;// restrict accessible elements to active length
		}



		void Init( ConstPtr const pdata, IndexType length )
		{
			this->m_pData		= (Ptr)pdata;
			this->m_Length		= length;
			this->m_AllocSize	= 0;
			this->m_Capacity		= length;// restrict accessible elements to active length
		}


		void Init( const MemoryBase<T, IndexType>& obj )
		{
			this->m_pData		= (T*)obj.begin();
			this->m_Length		= obj.Length();
			this->m_AllocSize	= 0;
			this->m_Capacity	= obj.Length();// restrict accessible elements to active length
		}


		void Init( std::initializer_list<T> ilist )
		{
			MemCopy( this->begin(), ilist.begin(), Min( ilist.size(), this->m_Length ) );
		}


		void Release()
		{
			this->m_pData		= nullptr;
			this->m_Length		= 0;
			this->m_Capacity	= 0;
		}


		void Display() const
		{
			tcout << typeid(*this).name() << _T("[ ") << this->m_Length << _T(" ]:\n" );

			for( IndexType i=0; i<this->m_Length; ++i )
				tcout << _T("  [") << i << _T("]: ") << this->m_pData[i] << tendl;

			tcout << tendl;
		}
		


	private:

		// Delete unnecessary parent methods
		bool Reserve( IndexType ) = delete;
bool Reallocate( IndexType ) = delete;
//		bool Resize( IndexType ) = delete;
//		bool Resize( IndexType, const T& ) = delete;
//		bool Extend( IndexType ) = delete;
//		bool Shrink( IndexType ) = delete;
		IndexType InsertBefore( IndexType ) = delete;
		IndexType InsertBefore( IndexType, const T& ) = delete;
		IndexType InsertBefore( IndexType, T&& ) = delete;
		IndexType InsertAfter( IndexType ) = delete;
		IndexType InsertAfter( IndexType, const T& ) = delete;
		IndexType InsertAfter( IndexType, T&& ) = delete;

		// Hide parent methods
		using MemoryBase<T, IndexType>::Init;
		using MemoryBase<T, IndexType>::Release;
		//using MemoryBase<T, IndexType>::Clear;
		using MemoryBase<T, IndexType>::Reserve;
using MemoryBase<T, IndexType>::Reallocate;
//using MemoryBase<T, IndexType>::Resize;
//using MemoryBase<T, IndexType>::Extend;
//using MemoryBase<T, IndexType>::Shrink;
		using MemoryBase<T, IndexType>::InsertBefore;
		using MemoryBase<T, IndexType>::InsertAfter;

	};


}// end of namespace


#endif // !ARRAY_VIEW_H
