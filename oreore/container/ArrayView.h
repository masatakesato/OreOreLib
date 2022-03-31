#ifndef ARRAY_VIEW_H
#define	ARRAY_VIEW_H

#include	"../common/TString.h"
//#include	"../memory/Memory.h"

#include	"ArrayBase.h"



// https://www.codeproject.com/Articles/848746/ArrayView-StringView



namespace OreOreLib
{


	template< typename T, typename InexType >
	class ArrayBase< detail::ARRVIEW<T>, detail::DynamicSize, InexType > : public Memory<T, InexType>
	{
		//using InexType = typename Memory<T, InexType>::InexType;
		using Ptr = T*;
		using ConstPtr = const T*;

	public:

		// Default constructor
		ArrayBase()
			: Memory<T, InexType>()
		{
		}


		// Constructor
		ArrayBase( ConstPtr const pdata, InexType length )
		{
			Init( pdata, length );
		}


		// Constructor
		ArrayBase( const Memory<T, InexType>& obj )
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



		void Init( ConstPtr const pdata, InexType length )
		{
			this->m_pData		= (Ptr)pdata;
			this->m_Length		= length;
			this->m_AllocSize	= 0;
			this->m_Capacity		= length;// restrict accessible elements to active length
		}


		void Init( const Memory<T, InexType>& obj )
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

			for( InexType i=0; i<this->m_Length; ++i )
				tcout << _T("  [") << i << _T("]: ") << this->m_pData[i] << tendl;

			tcout << tendl;
		}
		


	private:

		// Delete unnecessary parent methods
		bool Resize( InexType ) = delete;
		bool Resize( InexType, const T& ) = delete;
		bool Reserve( InexType ) = delete;
		bool Extend( InexType ) = delete;
		bool Shrink( InexType ) = delete;
		InexType InsertBefore( InexType ) = delete;
		InexType InsertBefore( InexType, const T& ) = delete;
		InexType InsertBefore( InexType, T&& ) = delete;
		InexType InsertAfter( InexType ) = delete;
		InexType InsertAfter( InexType, const T& ) = delete;
		InexType InsertAfter( InexType, T&& ) = delete;

		// Hide parent methods
		using Memory<T, InexType>::Init;
		using Memory<T, InexType>::Release;
		//using Memory<T, InexType>::Clear;
		using Memory<T, InexType>::Reserve;
		using Memory<T, InexType>::Resize;
		using Memory<T, InexType>::Extend;
		using Memory<T, InexType>::Shrink;
		using Memory<T, InexType>::InsertBefore;
		using Memory<T, InexType>::InsertAfter;

	};


}// end of namespace


#endif // !ARRAY_VIEW_H
