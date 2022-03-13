#ifndef ARRAY_VIEW_H
#define	ARRAY_VIEW_H

#include	"../common/TString.h"
//#include	"../memory/Memory.h"

#include	"ArrayBase.h"



// https://www.codeproject.com/Articles/848746/ArrayView-StringView



namespace OreOreLib
{


	template< typename T >
	class ArrayBase< detail::ARRVIEW<T>, detail::DynamicSize > : public Memory<T>
	{
		using SizeType = typename Memory<T>::SizeType;
		using Ptr = T*;
		using ConstPtr = const T*;

	public:

		// Default constructor
		ArrayBase()
			: Memory<T>()
		{
		}


		// Constructor
		ArrayBase( ConstPtr const pdata, SizeType length )
		{
			Init( pdata, length );
		}


		// Constructor
		ArrayBase( const Memory<T>& obj )
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



		void Init( ConstPtr const pdata, SizeType length )
		{
			this->m_pData		= (Ptr)pdata;
			this->m_Length		= length;
			this->m_AllocSize	= 0;
			this->m_Capacity		= length;// restrict accessible elements to active length
		}


		void Init( const Memory<T>& obj )
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

			for( SizeType i=0; i<this->m_Length; ++i )
				tcout << _T("  [") << i << _T("]: ") << this->m_pData[i] << tendl;

			tcout << tendl;
		}
		


	private:

		// Delete unnecessary parent methods
		bool Resize( SizeType ) = delete;
		bool Resize( SizeType, const T& ) = delete;
		bool Reserve( SizeType ) = delete;
		bool Extend( SizeType ) = delete;
		bool Shrink( SizeType ) = delete;
		SizeType InsertBefore( SizeType ) = delete;
		SizeType InsertBefore( SizeType, const T& ) = delete;
		SizeType InsertBefore( SizeType, T&& ) = delete;
		SizeType InsertAfter( SizeType ) = delete;
		SizeType InsertAfter( SizeType, const T& ) = delete;
		SizeType InsertAfter( SizeType, T&& ) = delete;

		// Hide parent methods
		using Memory<T>::Init;
		using Memory<T>::Release;
		//using Memory<T>::Clear;
		using Memory<T>::Reserve;
		using Memory<T>::Resize;
		using Memory<T>::Extend;
		using Memory<T>::Shrink;
		using Memory<T>::InsertBefore;
		using Memory<T>::InsertAfter;

	};


}// end of namespace


#endif // !ARRAY_VIEW_H
