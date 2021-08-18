#include	<crtdbg.h>
#include	<iostream>
using namespace std;



class Test
{
public:

	Test(){}
	~Test(){}



	// new
	void* operator new( size_t size )
	{
		cout << "Test:: operator new( size_t size )...\n";

		void* ptr = malloc( size );

		if( !ptr )
		{
			bad_alloc ba;
			cout << "Memory allocation error.\n";
			throw ba;
		}
		else
		{
			cout << "Memory is allocated successfully!\n";
			return ptr;
		}

	}

	// delete
	void operator delete( void* ptr ) noexcept
	{
		cout << "Test:: operator delete( void* ptr ) noexcept...\n";
		//cout << "Test:: Free the memory allocated by the delete operator.\n";
		free( ptr );
	}


	// placement new
	void* operator new( size_t size, void* p )
	{
		cout << "Test:: operator new( size_t size, void* p )...\n";
		void* ptr = p;//malloc( size );
		return ptr;
	}


	// placement delete
	void operator delete( void* ptr, void* ) noexcept
	{
		cout << "Test:: operator delete( void* ptr, void* ) noexcept...\n";
//		free( ptr );
	}

};





void* operator new( size_t size )
{
	cout << "new: "<< size << endl;
	return malloc( size );
}



void operator delete( void* p, size_t size ) noexcept
{
	cout << "delete: " << size << endl;
	free(p);
}



void* operator new[]( size_t size )
{
	cout << "new[]: "<< size << endl;
	return malloc( size );
}



void operator delete[]( void* p, size_t size ) noexcept
{
	cout << "delete[]: " << size << endl;
	free(p);
}





int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	int *a = new int(0);
	delete a;

	int *arr = new int[1000];
	delete [] arr;


	unsigned char *carr = new unsigned char[ sizeof(Test) * 3 ];

//	unsigned char *pchar = new (carr) unsigned char;
//	carr[0] = 6;
//	cout << (int)*(unsigned char*)pchar << endl;

	Test* t = new (carr) Test();
	t->~Test();


	delete [] carr;


	return 0;
}