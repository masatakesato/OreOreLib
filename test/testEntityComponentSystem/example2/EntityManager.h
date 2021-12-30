#ifndef ENTITY_MANAGER_H
#define	ENTITY_MANAGER_H

#include	<oreore/container/StaticArray.h>
#include	<oreore/container/RingQueue.h>


#include	"ECSCommon.h"



class EntityManager
{
public:

	EntityManager()
	{
		// キューをIDで初期化する
		for( Entity entity=0; entity<MAX_ENTITIES; ++entity )
			mAvailableEntities.Enqueue( entity );
	}


	Entity CreateEntity()
	{
		ASSERT( m_NumActiveEntities < MAX_ENTITIES && _T("Too many entites exist.") );

		// キューからIDを取り出す
		Entity id = mAvailableEntities.Dequeue();
		++m_NumActiveEntities;

		return id;
	}


	void DestroyEntity( Entity entity )
	{
		ASSERT( entity < MAX_ENTITIES && _T("Entity out of range.") );

		// Signatureを全てゼロにリセットする
		mSignatures[ entity ].UnsetAll();

		// 解放したエンティティIDをmAvailableEntitiesに戻す
		mAvailableEntities.Enqueue( entity );
		--m_NumActiveEntities;

	}


	void SetSignature( Entity entity, Signature signature )
	{
		ASSERT( entity < MAX_ENTITIES && _T("Entity out of range.") );

		mSignatures[ entity ].CopyFrom( &signature );
	}


	Signature GetSignature( Entity entity )
	{
		ASSERT( entity < MAX_ENTITIES && _T("Entity out of range.") );

		return mSignatures[ entity ];
	}



private:

	// Queue of unused entity IDs
	OreOreLib::RingQueue<Entity>	mAvailableEntities{};

	// Array of signatures where the index corresponds to the entity ID
	OreOreLib::StaticArray<Signature, MAX_ENTITIES>	mSignatures{};

	// Total living entities - used to keep limits on how many exist
	uint32	m_NumActiveEntities{};
};



#endif // !ENTITY_MANAGER_H


