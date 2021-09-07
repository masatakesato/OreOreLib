#include	<crtdbg.h>

#include	<oreore/common/TString.h>
#include	<oreore/thread/IRunnable.h>
#include	<oreore/thread/Thread.h>


class MyFunc : public OreOreLib::IRunnable
{
public:

	MyFunc( int& val1, int &val2 )
		: m_val1( val1 )
		, m_val2( val2 )
	{
		
	}



	void IRunnable::Run()
	{
		tcout << "//============= MyFunc::Run() ==============//\n";

		for( int i=0; i<1000; ++i)
		{
		m_val1++;
		m_val2--;
		}

//		std::chrono::milliseconds dura( 2000 );
//		std::this_thread::sleep_for( dura );
	}


private:

	int& m_val1;
	int& m_val2;


};



int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	OreOreLib::Thread thread1;

	int a = 0;
	int b = 0;


	while(1)
	{
		MyFunc* m = new MyFunc( a, b );

	thread1.Init( /*new MyFunc( a, b )*/m );
	//thread1.Init( a );
	//thread1.Play();


	tcout << a << tendl;
	tcout << b << tendl;


	SafeDelete(m);

	}

	return 0;
}