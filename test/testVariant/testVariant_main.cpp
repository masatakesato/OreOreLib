#include	<crtdbg.h>

#include	<oreore/common/TString.h>
#include	<oreore/memory/SharedPtr.h>
#include	<oreore/memory/WeakPtr.h>
#include	<oreore/container/Variant2.h>
using namespace OreOreLib;


#include	<memory>
#include	<string>
using namespace std;





int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	//SharedPtr<int> sp( new int(5) );
	////shared_ptr<int> sp( new int(5) );

	//void* valaa = sp;



	//tcout << valaa << tendl;

	//return 0;



	Variant2 var;

	//while(1)
	{
		var = string("Hello World");
		tcout << ((string)var).c_str() << tendl;


		auto tmp_str = string("Hello World2");
		var = (string&&)tmp_str;
		tcout << ((string)var).c_str() << tendl;


		var = nullptr;

	
		SharedPtr<int> ival = new int(6);//shared_ptr<int> ival( new int(6) );
		var = ival;//(SharedPtr<int>)ival;
		tcout << *(SharedPtr<int>)var << tendl;


		SharedPtr<int> ival2 = new int(-222);
		var = ival2;
		tcout << *(SharedPtr<int>)var << tendl;


		//SharedPtr<int> ival3 = new int(-222);
		var = SharedPtr<int>( new int(-222) );//ival3;
		tcout << *(SharedPtr<int>)var << tendl;

		
		var = 6.5666f;
		tcout << float(var) << tendl;


		var = WeakPtr<int>(ival2  );//dval1;
		tcout << *(WeakPtr<int>)var << tendl;

	}

//	tcout << *SharedPtr<int>(var) << tendl;


	Variant2 var2 = SharedPtr<int>( new int(888888888) );//6.5666f;
	tcout << *(SharedPtr<int>)var2 << tendl;

	return 0;
}