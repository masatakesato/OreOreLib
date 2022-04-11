#ifndef MEMORY_OPERATIONS_H
#define	MEMORY_OPERATIONS_H

//#include	<type_traits>

#include	"../common/Utility.h"



namespace OreOreLib
{
	namespace Mem
	{


		//##############################################################################################//
		//																								//
		//										Helper functions										//
		//																								//
		//##############################################################################################//

		namespace detail
		{

			template < class F, class SrcIter, class DstIter, class... Params >
			void ForwardIteration( F&& func, DstIter* pDst, const SrcIter* pSrc, sizeType size, Params... params )
			{
				SrcIter* begin = (SrcIter*)pSrc;
				const SrcIter* end = pSrc + size;
				DstIter* out = pDst;

				while( begin != end )
				{
					func( out, begin, params... );
					++begin; ++out;
				}
			}



			template < class F, class SrcIter, class DstIter, class... Params >
			void BackwardIteration( F&& func, DstIter* pDst, const SrcIter* pSrc, sizeType size, Params...params )
			{
				SrcIter* begin = (SrcIter*)pSrc + size - 1;
				const SrcIter* end = pSrc - 1;
				DstIter* out = pDst + size - 1;

				while( begin != end )
				{
					func( out, begin, params... );
					--begin; --out;
				}
			}



			template < class SrcIter, class DstIter >
			const auto CopyOp = []( DstIter* dst, SrcIter* src )
			{
				//tcout << "CopyOp()...\n";
				dst->~DstIter();// Destruct existing data from destination memory
				new ( dst ) DstIter( *(DstIter*)src );// Call copy constructor

				// Copy assignment operator version
				//*dst = *(DstIter*)src;
			};


			template < class SrcIter, class DstIter >
			const auto UninitializedCopyOp = []( DstIter* dst, SrcIter* src )
			{
				//tcout << "UninitializedCopyOp()...\n";
				new ( dst ) DstIter( *(DstIter*)src );// Overwite existing memory with placement new
			};


			template < class SrcIter, class DstIter >
			const auto MigrateOp = []( DstIter* dst, SrcIter* src )//, bool cleanupSrc )
			{
				//tcout << "MigrateOp()...\n";
				dst->~DstIter();// Destruct existing data from dst
				new ( dst ) DstIter( (DstIter&&)( *src ) );// Overwite existing memory with placement new
				//if( cleanupSrc )	src->~SrcIter();// Destruct src
			};


			template < class SrcIter, class DstIter >
			const auto UninitializedMigrateOp = []( DstIter* dst, SrcIter* src )//, bool cleanupSrc )
			{
				//tcout << "UninitializedMigrateOp()...\n";
				new ( dst ) DstIter( (DstIter&&)( *src ) );// Overwite existing memory with placement new
				//if( cleanupSrc )	src->~SrcIter();// Destruct src
			};


		}// end of namespace detail



		//##############################################################################################//
		//																								//
		//							Copy / UninitializedCopy (memcpy equivalent)						//
		//																								//
		//##############################################################################################//

		#if __cplusplus >= 201703L	// Above C++17

		// If using Visual c++, folowing command must be added for __cplusplus macro activation.
		//   /Zc:__cplusplus

		template < class SrcIter, class DstIter >
		DstIter* Copy( DstIter* pDst, const SrcIter* pSrc, sizeType size )
		{
			if constexpr( std::is_same_v<SrcIter, DstIter> && std::is_trivially_copyable_v<SrcIter> )
			{
				return (DstIter*)memcpy( pDst, pSrc, sizeof DstIter * size );
			}
			else
			{
				detail::ForwardIteration( detail::CopyOp<SrcIter, DstIter>, pDst, pSrc, size );
				return pDst;
			}
		}



		template < class SrcIter, class DstIter >
		DstIter* UninitializedCopy( DstIter* pDst, const SrcIter* pSrc, sizeType size )
		{
			if constexpr( std::is_same_v<SrcIter, DstIter> && std::is_trivially_copyable_v<SrcIter> )
			{
				return (DstIter*)memcpy( pDst, pSrc, sizeof DstIter * size );
			}
			else
			{
				detail::ForwardIteration( detail::UninitializedCopyOp<SrcIter, DstIter>, pDst, pSrc, size );
				return pDst;
			}
		}



		#else	// Below C++14

		//======================== Trivial ======================//

		template < class Iter >
		std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
		Copy( Iter* pDst, const Iter* pSrc, sizeType size )
		{
			return (Iter*)memcpy( pDst, pSrc, sizeof Iter * size );
		}

		template < class Iter >
		std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
		UninitializedCopy( Iter* pDst, const Iter* pSrc, sizeType size )
		{
			return (Iter*)memcpy( pDst, pSrc, sizeof Iter * size );
		}


		//====================== Non-Trivial ====================//

		template < class SrcIter, class DstIter >
		DstIter* Copy( DstIter* pDst, const SrcIter* pSrc, sizeType size )
		{
			detail::ForwardIteration( detail::CopyOp<SrcIter, DstIter>, pDst, pSrc, size );
			return pDst;
		}

		template < class SrcIter, class DstIter >
		DstIter* UninitializedCopy( DstIter* pDst, const SrcIter* pSrc, sizeType size )
		{
			detail::ForwardIteration( detail::UninitializedCopyOp<SrcIter, DstIter>, pDst, pSrc, size );
			return pDst;
		}



		#endif




		//##############################################################################################//
		//																								//
		//						SafeCopy / Uninitialized_SafeCopy (memmove equivalent)					//
		//																								//
		//##############################################################################################//

		#if __cplusplus >= 201703L	// Above C++17

		template < class SrcIter, class DstIter >
		DstIter* SafeCopy( DstIter* pDst, const SrcIter* pSrc, sizeType size )
		{
			if constexpr( std::is_same_v<SrcIter, DstIter> && std::is_trivially_copyable_v<SrcIter> )
			{
				return (DstIter*)memmove( pDst, pSrc, sizeof DstIter * size );
			}
			else
			{
				if( pSrc < pDst )// Copy from the last element
				{
					detail::BackwardIteration( detail::CopyOp<SrcIter, DstIter>, pDst, pSrc, size );
				}
				else if( pSrc > pDst )// Copy from the first element
				{
					detail::ForwardIteration( detail::CopyOp<SrcIter, DstIter>, pDst, pSrc, size );
				}

				return pDst;
			}
		}



		template < class SrcIter, class DstIter >
		DstIter* UninitializedSafeCopy( DstIter* pDst, const SrcIter* pSrc, sizeType size )
		{
			if constexpr( std::is_same_v<SrcIter, DstIter> && std::is_trivially_copyable_v<SrcIter> )
			{
				return (DstIter*)memmove( pDst, pSrc, sizeof DstIter * size );
			}
			else
			{
				if( pSrc < pDst )// Copy from the last element
				{
					detail::BackwardIteration( detail::UninitializedCopyOp<SrcIter, DstIter>, pDst, pSrc, size );
				}
				else if( pSrc > pDst )// Copy from the first element
				{
					detail::ForwardIteration( detail::UninitializedCopyOp<SrcIter, DstIter>, pDst, pSrc, size );
				}

				return pDst;
			}
		}



		#else	// Below C++14

		//======================== Trivial ======================//

		template < class Iter >
		std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
		SafeCopy( Iter* pDst, const Iter* pSrc, sizeType size )
		{
			return (Iter*)memmove( pDst, pSrc, sizeof Iter * size );
		}

		template < class Iter >
		std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
		UninitializedSafeCopy( Iter* pDst, const Iter* pSrc, sizeType size )
		{
			return (Iter*)memmove( pDst, pSrc, sizeof Iter * size );
		}


		//====================== Non-Trivial ====================//

		template < class SrcIter, class DstIter >
		DstIter* SafeCopy( DstIter* pDst, const SrcIter* pSrc, sizeType size )
		{
			if( pSrc < pDst )// Copy from the last element
			{
				detail::BackwardIteration( detail::CopyOp<SrcIter, DstIter>, pDst, pSrc, size );
			}
			else if( pSrc > pDst )// Copy from the first element
			{
				detail::ForwardIteration( detail::CopyOp<SrcIter, DstIter>, pDst, pSrc, size );
			}

			return pDst;
		}

		template < class SrcIter, class DstIter >
		DstIter* UninitializedSafeCopy( DstIter* pDst, const SrcIter* pSrc, sizeType size )
		{
			if( pSrc < pDst )// Copy from the last element
			{
				detail::BackwardIteration( detail::UninitializedCopyOp<SrcIter, DstIter>, pDst, pSrc, size );
			}
			else if( pSrc > pDst )// Copy from the first element
			{
				detail::ForwardIteration( detail::UninitializedCopyOp<SrcIter, DstIter>, pDst, pSrc, size );
			}

			return pDst;
		}



		#endif



		//##############################################################################################//
		//																								//
		//								Migrate / UninitializedMigrate									//
		//																								//
		//##############################################################################################//

		#if __cplusplus >= 201703L	// Above C++17

		template < class SrcIter, class DstIter >
		DstIter* Migrate( DstIter* pDst, SrcIter* pSrc, sizeType size )
		{
			if constexpr( std::is_same_v<SrcIter, DstIter> && std::is_trivially_copyable_v<SrcIter> )
			{
				auto result = (DstIter*)memmove( pDst, pSrc, sizeof DstIter * size );
				memset( pSrc, 0, sizeof SrcIter * size );
				return result;
			}
			else
			{
				detail::ForwardIteration( detail::MigrateOp<SrcIter, DstIter>, pDst, pSrc, size );
				return pDst;
			}
		}



		template < class SrcIter, class DstIter >
		DstIter* UninitializedMigrate( DstIter* pDst, SrcIter* pSrc, sizeType size )
		{
			if constexpr( std::is_same_v<SrcIter, DstIter> && std::is_trivially_copyable_v<SrcIter> )
			{
				auto result = (DstIter*)memmove( pDst, pSrc, sizeof DstIter * size );
				memset( pSrc, 0, sizeof SrcIter * size );
				return result;
			}
			else
			{
				detail::ForwardIteration( detail::UninitializedMigrateOp<SrcIter, DstIter>, pDst, pSrc, size );
				return pDst;
			}
		}


		#else	// Below C++14

		//======================== Trivial ======================//

		template < class Iter >
		std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
		Migrate( Iter* pDst, Iter* pSrc, sizeType size )
		{
			auto result = (Iter*)memcpy( pDst, pSrc, sizeof Iter * size );
			memset( pSrc, 0, sizeof Iter * size );
			return result;
		}

		template < class Iter >
		std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
		UninitializedMigrate( Iter* pDst, Iter* pSrc, sizeType size )
		{
			auto result = (Iter*)memcpy( pDst, pSrc, sizeof Iter * size );
			memset( pSrc, 0, sizeof Iter * size );
			return result;
		}


		//====================== Non-Trivial ====================//

		template < class SrcIter, class DstIter >
		DstIter* Migrate( DstIter* pDst, SrcIter* pSrc, sizeType size )
		{
			detail::ForwardIteration( detail::MigrateOp<SrcIter, DstIter>, pDst, pSrc, size );
			return pDst;
		}

		template < class SrcIter, class DstIter >
		DstIter* UninitializedMigrate( DstIter* pDst, SrcIter* pSrc, sizeType size )
		{
			detail::ForwardIteration( detail::UninitializedMigrateOp<SrcIter, DstIter>, pDst, pSrc, size );
			return pDst;
		}



		#endif




		//##############################################################################################//
		//																								//
		//							SafeMigrate / UninitializedSafeMigrate								//
		//																								//
		//##############################################################################################//

		#if __cplusplus >= 201703L	// Above C++17

		template < class SrcIter, class DstIter >
		DstIter* SafeMigrate( DstIter* pDst, SrcIter* pSrc, sizeType size )
		{
			if constexpr( std::is_same_v<SrcIter, DstIter> && std::is_trivially_copyable_v<SrcIter> )
			{
				auto result = (DstIter*)memmove( pDst, pSrc, sizeof DstIter * size );
				memset( pSrc, 0, sizeof SrcIter * size );
				return result;
			}
			else
			{
				if( pSrc < pDst )// Migrate from the last element
				{
					detail::BackwardIteration( detail::MigrateOp<SrcIter, DstIter>, pDst, pSrc, size );
				}
				else if( pSrc > pDst )// Migrate from the first element
				{
					detail::ForwardIteration( detail::MigrateOp<SrcIter, DstIter>, pDst, pSrc, size );
				}

				return pDst;
			}
		}



		template < class SrcIter, class DstIter >
		DstIter* UninitializedSafeMigrate( DstIter* pDst, SrcIter* pSrc, sizeType size )
		{
			if constexpr( std::is_same_v<SrcIter, DstIter> && std::is_trivially_copyable_v<SrcIter> )
			{
				auto result = (DstIter*)memmove( pDst, pSrc, sizeof DstIter * size );
				memset( pSrc, 0, sizeof SrcIter * size );
				return result;
			}
			else
			{
				if( pSrc < pDst )// Migrate from the last element
				{
					detail::BackwardIteration( detail::UninitializedMigrateOp<SrcIter, DstIter>, pDst, pSrc, size );
				}
				else if( pSrc > pDst )// Migrate from the first element
				{
					detail::ForwardIteration( detail::UninitializedMigrateOp<SrcIter, DstIter>, pDst, pSrc, size );
				}

				return pDst;
			}
		}


		#else	// Below C++14

		//======================== Trivial ======================//

		template < class Iter >
		std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
		SafeMigrate( Iter* pDst, Iter* pSrc, sizeType size )
		{
			auto result = (Iter*)memmove( pDst, pSrc, sizeof Iter * size );
			memset( pSrc, 0, sizeof Iter * size );
			return result;
		}

		template < class Iter >
		std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
		UninitializedSafeMigrate( Iter* pDst, Iter* pSrc, sizeType size )
		{
			auto result = (Iter*)memmove( pDst, pSrc, sizeof Iter * size );
			memset( pSrc, 0, sizeof Iter * size );
			return result;
		}


		//====================== Non-Trivial ====================//

		template < class SrcIter, class DstIter >
		DstIter* SafeMigrate( DstIter* pDst, SrcIter* pSrc, sizeType size )
		{
			if( pSrc < pDst )// Migrate from the last element
			{
				detail::BackwardIteration( detail::MigrateOp<SrcIter, DstIter>, pDst, pSrc, size );
			}
			else if( pSrc > pDst )// Migrate from the first element
			{
				detail::ForwardIteration( detail::MigrateOp<SrcIter, DstIter>, pDst, pSrc, size );
			}

			return pDst;
		}

		template < class SrcIter, class DstIter >
		DstIter* UninitializedSafeMigrate( DstIter* pDst, SrcIter* pSrc, sizeType size )
		{
			if( pSrc < pDst )// Migrate from the last element
			{
				detail::BackwardIteration( detail::UninitializedMigrateOp<SrcIter, DstIter>, pDst, pSrc, size );
			}
			else if( pSrc > pDst )// Migrate from the first element
			{
				detail::ForwardIteration( detail::UninitializedMigrateOp<SrcIter, DstIter>, pDst, pSrc, size );
			}

			return pDst;
		}



		#endif




		//##############################################################################################//
		//																								//
		//									Init / UninitializedInit									//
		//																								//
		//##############################################################################################//

		#if __cplusplus >= 201703L	// Above C++17

		template < class Iter >
		Iter* Init( Iter* pDst, sizeType size )
		{
			if constexpr( std::is_trivially_copyable_v<Iter> )
			{
				return (Iter*)memset( pDst, 0, sizeof Iter * size );
			}
			else
			{
				Iter* begin = pDst;
				const Iter* end = pDst + size;

				while( begin != end )
				{
					begin->~Iter();// Desctuct existing data
					new ( begin ) Iter();
					++begin;
				}

				return pDst;
			}
		}


		template < class Iter >
		Iter* UninitializedInit( Iter* pDst, sizeType size )
		{
			if constexpr( std::is_trivially_copyable_v<Iter> )
			{
				return (Iter*)memset( pDst, 0, sizeof Iter * size );
			}
			else
			{
				Iter* begin = pDst;
				const Iter* end = pDst + size;

				while( begin != end )
				{
					new ( begin ) Iter();
					++begin;
				}

				return pDst;
			}
		}



		#else	// Below C++14

		//====================== Non-Trivial ====================//

		template < class Iter >
		std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
		Init( Iter* pDst, sizeType size )
		{
			return (Iter*)memset( pDst, 0, sizeof Iter * size );
		}

		template < class Iter >
		std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
		UninitializedInit( Iter* pDst, sizeType size )
		{
			return (Iter*)memset( pDst, 0, sizeof Iter * size );
		}



		//====================== Non-Trivial ====================//

		template < class Iter >
		std::enable_if_t< !std::is_trivially_copyable_v<Iter>, Iter* >
		Init( Iter* pDst, sizeType size )
		{
			Iter* begin = pDst;
			const Iter* end = pDst + size;

			while( begin != end )
			{
				begin->~Iter();// Desctuct existing data
				new ( begin ) Iter();
				++begin;
			}

			return pDst;
		}

		template < class Iter >
		std::enable_if_t< !std::is_trivially_copyable_v<Iter>, Iter* >
		UninitializedInit( Iter* pDst, sizeType size )
		{
			Iter* begin = pDst;
			const Iter* end = pDst + size;

			while( begin != end )
			{
				new ( begin ) Iter();
				++begin;
			}

			return pDst;
		}

		#endif




		//##############################################################################################//
		//																								//
		//												Clear											//
		//																								//
		//##############################################################################################//

		#if __cplusplus >= 201703L	// Above C++17

		template < class Iter >
		Iter* Clear( Iter* pDst, sizeType size )
		{
			if constexpr( std::is_trivially_copyable_v<Iter> )
			{
				return (Iter*)memset( pDst, 0, sizeof Iter * size );
			}
			else
			{
				Iter* begin = pDst;
				const Iter* end = pDst + size;

				while( begin != end )
				{
					begin->~Iter();// Desctuct existing data
					++begin;
				}

				return pDst;
			}
		}



		#else	// Below C++14

		//====================== Non-Trivial ====================//

		template < class Iter >
		std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
		Clear( Iter* pDst, sizeType size )
		{
			return (Iter*)memset( pDst, 0, sizeof Iter * size );
		}



		//====================== Non-Trivial ====================//

		template < class Iter >
		std::enable_if_t< !std::is_trivially_copyable_v<Iter>, Iter* >
		Clear( Iter* pDst, sizeType size )
		{
			Iter* begin = pDst;
			const Iter* end = pDst + size;

			while( begin != end )
			{
				begin->~Iter();// Desctuct existing data
				++begin;
			}

			return pDst;
		}


		#endif



	}// end of namespace Mem

}// end of namespace OreOreLib


#endif // !MEMORY_OPERATIONS_H
