#ifndef SYSTEM_MANAGER_H
#define	SYSTEM_MANAGER_H

#include	<oreore/container/HashMap.h>

#include	"ECSCommon.h"
#include	"System.h"



class SystemManager
{
public:

	template < typename T >
	T* RegisterSystem()
	{
		const char* typeName = typeid(T).name();

		ASSERT( !mSystems.Exists( typeName ) && "System already exists." );

		auto system = new T();
		mSystems.Put( typeName, system );

		return system;
	}


	template < typename T >
	void SetSignature( Signature signature )
	{
		const char* typeName = typeid(T).name();

		ASSERT( mSystems.Exists( typeName ) && "System does not exist." );

		mSignatures.Put( typeName, signature );
	}


	void EntityDestroyed( Entity entity )
	{
		for( const auto& pair : mSystems )
		{
			const auto& system = pair.second;
			system->mEntities.Remove( entity );
		}
	}


	void EntitySignatureChanged( Entity entity, Signature entitySignature )
	{
		for( const auto& pair : mSystems )
		{
			const auto& type = pair.first;
			const auto& system = pair.second;
			const auto& systemSignature = mSignatures[ type ];

			if( ( entitySignature & systemSignature ) == systemSignature )
			{
				system->mEntities.Put( entity );
			}
			else
			{
				system->mEntities.Remove( entity );
			}
		}
	}



private:

	OreOreLib::HashMap<const char*, Signature, 128>	mSignatures{};

	OreOreLib::HashMap<const char*, System*, 128>	mSystems;

};



#endif // !SYSTEM_MANAGER_H
