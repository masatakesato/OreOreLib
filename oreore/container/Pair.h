#ifndef PAIR_H
#define	PAIR_H


template< typename T1, typename T2 >
struct Pair
{
	// Default constructor
	Pair()
		: first( T1() )
		, second( T2() )
	{

	}


	// Constructor
	Pair( const T1& x )
		: first( x )
		, second( T2() )
	{
		//tcout << "Pair( const T1& x )\n";
	}


	// Constructor
	Pair( const T1& x, const T2& y )
		: first( x )
		, second( y )
	{
		//tcout << "Pair( const T1& x, const T2& y )\n";
	}


	// Constructor with move assignment
	Pair( T1&& x, T2&& y )
		: first( std::forward<T1>(x) )
		, second( std::forward<T2>(y) )
	{
		//tcout << "Pair( T1&& x, T2&& y )\n";
	}


	// Copy constructor
	Pair( const Pair& obj )
		: first( obj.first )
		, second( obj.second )
	{
//		tcout << "Pair( const Pair& obj )\n";
	}


	// Move constructor
	Pair( Pair&& obj )
		: first( std::forward<T1>(obj.first) )
		, second( std::forward<T1>(obj.second) )
	{

	}


	// Destructor
	~Pair()
	{

	}


	// Copy assignment operator
	Pair& operator=( const Pair& obj )
	{
		if( this != &obj )
		{
			first	= obj.first;
			second	= obj.second;
		}

		return *this;
	}


	// Move assignment operator
	Pair& operator=( Pair&& obj )
	{
		if( this != &obj )
		{
			first	= std::forward<T1>( obj.first );
			second	= std::forward<T1>( obj.second );
		}

		return *this;
	}



	T1 first;
	T2 second;

};



#endif // !PAIR_H
