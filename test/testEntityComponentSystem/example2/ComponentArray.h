#ifndef COMPONENT_ARRAY_H
#define	COMPONENT_ARRAY_H

#include	<oreore/container/StaticArray.h>
#include	<oreore/container/HashMap.h>

#include	"ECSCommon.h"



class IComponentArray
{
public:

	virtual ~IComponentArray() = default;
	virtual void EntityDestroyed( Entity entity ) = 0;
};



template < typename T >
class ComponentArray : public IComponentArray
{
public:

	void InsertData( Entity entity, T component )
	{
		ASSERT( !mEntityToIndexMap.Exists(entity) && "Component already exists." );

		size_t newIndex = mSize;
		mEntityToIndexMap[ entity ] = newIndex;
		mIndexToEntityMap[ newIndex ] = entity;
		mComponentArray[ newIndex ] = component;
		++mSize;
	}


	void RemoveData( Entity entity )
	{
		ASSERT( mEntityToIndexMap.Exists(entity) && "Component does not exist." );

		// 削除予定位置の要素を最終要素で上書きする
		size_t indexOfRemovedEntity = mEntityToIndexMap[ entity ];
		size_t indexOfLastElement = mSize - 1;
		mComponentArray[ indexOfRemovedEntity ] = mComponentArray[ indexOfLastElement ];

		// 
		Entity entityOfLastElement = mIndexToEntityMap[ indexOfLastElement ];
		mEntityToIndexMap[ entityOfLastElement ] = indexOfRemovedEntity;// 最終要素Entity -> 空いた要素番号
		mIndexToEntityMap[ indexOfRemovedEntity ] = entityOfLastElement;// 空いた要素番号 -> 最終要素Entity

		mEntityToIndexMap.Remove( entity );
		mIndexToEntityMap.Remove( indexOfLastElement );

		--mSize;
	}


	T& GetData( Entity entity )
	{
		ASSERT( mEntityToIndexMap.Exists(entity) && "Component does not exist." );

		return mComponentArray[ mEntityToIndexMap[entity] ];
	}


	void EntityDestroyed( Entity entity ) override
	{
		if( mEntityToIndexMap.Exists(entity) )
			RemoveData( entity );
	}



private:

	// Entityが使用するコンポーネントデータを格納するメモリ. 隙間だらけになってる可能性あり
	OreOreLib::StaticArray<T, MAX_ENTITIES>	mComponentArray;

	// エンティティIDからmComponentArrayインデックスへの変換テーブル
	OreOreLib::HashMap<Entity, size_t, 128>	mEntityToIndexMap;

	// 通し番号で利用中の要素を検索するためのインデックステーブル
	OreOreLib::HashMap<Entity, size_t, 128>	mIndexToEntityMap;

	size_t	mSize;
};


#endif // !COMPONENT_ARRAY_H
