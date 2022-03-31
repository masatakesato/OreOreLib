#ifndef HASH_CODE_H
#define	HASH_CODE_H

#include	"Utility.h"


// https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1178/lectures/27-Inheritance/code/Inheritance/lib/StanfordCPPLib/collections/hashcode.h
// https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1178/lectures/27-Inheritance/code/Inheritance/lib/StanfordCPPLib/collections/hashcode.cpp




namespace OreOreLib
{


	namespace HashConst
	{
		const uint64 DefaultHashSize = 16;
		const uint64 Seed		= 5381;
		const uint64 Multiplier	= 33;//31
		const uint64 Mask		= uint64(-1) >> 1;
	}




	template < typename T >
	std::enable_if_t< std::is_integral_v<T>, uint64 >
	HashCode( const T& key )
	{
		return key & HashConst::Mask;
	}



	template < typename T >
	std::enable_if_t< std::is_floating_point_v<T>, uint64 >
	HashCode( const T& key )
	{
		uint8* byte = (uint8*)&key;
		uint64 hash = HashConst::Seed;
		for( int i=0; i<sizeof(T); ++i )
			hash = HashConst::Multiplier * hash + (uint64) *byte++;

		return hash & HashConst::Mask;
	}



	template < typename T >
	std::enable_if_t< std::is_same_v<T,tstring> || std::is_same_v<T,std::string>, uint64 >
	HashCode( const T& key )
	{
		unsigned hash = HashConst::Seed;
		for( int i=0; i<key.length(); ++i )
			hash = HashConst::Multiplier * hash + key[i];

		return uint64( hash & HashConst::Mask );
	}



	template < typename T >
	std::enable_if_t< std::is_pointer_v<T>, uint64 >
	HashCode( const T key )
	{
		return HashCode( uint64(key) );
	}



	template < typename K, size_t TableSize=-1 >	struct KeyHash;


	template < typename K, size_t TableSize >
	struct KeyHash
	{
		uint64 operator()( const K& key ) const
		{
			return HashCode( key ) % TableSize;
			//return *(uint64*)( &key ) % TableSize;
		}


		template < typename T >
		T Get( const K& key ) const
		{
			return (T)HashCode( key ) % TableSize;
		}

	};



	template < typename K >
	struct KeyHash<K, -1>
	{
		uint64 operator()( const K& key, size_t tableSize ) const
		{
			return HashCode( key ) % tableSize;
			//return *(uint64*)( &key ) % TableSize;
		}


		template < typename T >
		T Get( const K& key, size_t tableSize ) const
		{
			return (T)HashCode( key ) % tableSize;
		}

	};




//
//
//
//static const int HASH_SEED = 5381;               // Starting point for first cycle
//static const int HASH_MULTIPLIER = 33;           // Multiplier for each cycle
//static const int HASH_MASK = unsigned(-1) >> 1;  // All 1 bits except the sign
//
//int hashCode(double key) {
//    char* byte = (char*) &key;
//    unsigned hash = HASH_SEED;
//    for (int i = 0; i < (int) sizeof(double); i++) {
//        hash = HASH_MULTIPLIER * hash + (int) *byte++;
//    }
//    return hash & HASH_MASK;
//}
//
//int hashCode(float key) {
//    char* byte = (char*) &key;
//    unsigned hash = HASH_SEED;
//    for (int i = 0; i < (int) sizeof(float); i++) {
//        hash = HASH_MULTIPLIER * hash + (int) *byte++;
//    }
//    return hash & HASH_MASK;
//}




}// end of namespace


#endif // !HASH_CODE_H
