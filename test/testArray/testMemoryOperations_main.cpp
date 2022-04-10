#include	<oreore/memory/MemoryOperations.h>
using namespace OreOreLib;


struct Val
{
	Val() : value( new float(0) )
	{
		tcout << "Val()\n";
	}

	Val( float val )
		: value( new float(val) )
	{
		tcout << "Val( float val )\n";
	}

	~Val()
	{
		tcout << "~Val(): " << tendl;
		SafeDelete( value );
	}

	Val( const Val& obj )
		: value( new float(*obj.value)  )
	{
		tcout << "Val( const Val& obj ) : " << *value << tendl;
	}

	Val( Val&& obj )
		: value( obj.value )
	{
		tcout << "Val( Val&& obj )\n";
		 obj.value = nullptr;
	}


	float* value;

};





int main()
{

	{
		tcout << _T("//============ Copy / UninitializedCopy ============//\n");
		std::string src[4]{"0", "1", "2", "3"}, dst[4];

		Copy( &dst[0], &src[0], 4 );
		//UninitializedCopy( &dst[0], &src[0], 4 );

		for( auto& val : dst )
			tcout << val.c_str() << tendl;
	}

	tcout << tendl;

	{
		tcout << _T("//============ SafeCopy / UninitializedSafeCopy ============//\n");
		std::string src[4]{"0", "1", "2", "3"};

		tcout << _T("-------------------------\n");

		tcout << _T("initial src:\n");
		for( auto& val : src )
			tcout << val.c_str() << tendl;

		//SafeCopy( &src[1], &src[0], 3 );
		UninitializedSafeCopy( &src[1], &src[0], 3 );

		tcout << _T("-------------------------\n");

		tcout << _T("result src:\n");
		for( auto& val : src )
			tcout << val.c_str() << tendl;
	}

	tcout << tendl;

	{
		tcout << _T("//============ Migrate / UninitializedMigrate ============//\n");
		std::string src[4]{"0", "1", "2", "3"}, dst[4];

		tcout << _T("-------------------------\n");

		tcout << _T("initial src:\n");
		for( auto& val : src )
			tcout << val.c_str() << tendl;

		tcout << _T("initial dst:\n");
		for( auto& val : dst )
			tcout << val.c_str() << tendl;

		//Migrate( &dst[0], &src[0], 4 );
		UninitializedMigrate( &dst[0], &src[0], 4 );

		tcout << _T("-------------------------\n");

		tcout << _T("result src:\n");
		for( auto& val : src )
			tcout << val.c_str() << tendl;

		tcout << _T("result dst:\n");
		for( auto& val : dst )
			tcout << val.c_str() << tendl;
	}

	tcout << tendl;

	{
		tcout << _T("//============ SafeMigrate / UninitializedSafeMigrate ============//\n");
		std::string src[4]{"0", "1", "2", "3"};

		tcout << _T("-------------------------\n");

		tcout << _T("initial src:\n");
		for( auto& val : src )
			tcout << val.c_str() << tendl;

		SafeMigrate( &src[1], &src[0], 3 );
		//UninitializedSafeMigrate( &src[1], &src[0], 3 );

		tcout << _T("-------------------------\n");

		tcout << _T("result src:\n");
		for( auto& val : src )
			tcout << val.c_str() << tendl;
	}



	return 0;
}