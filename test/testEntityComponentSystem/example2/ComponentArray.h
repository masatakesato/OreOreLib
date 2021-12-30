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

		// �폜�\��ʒu�̗v�f���ŏI�v�f�ŏ㏑������
		size_t indexOfRemovedEntity = mEntityToIndexMap[ entity ];
		size_t indexOfLastElement = mSize - 1;
		mComponentArray[ indexOfRemovedEntity ] = mComponentArray[ indexOfLastElement ];

		// 
		Entity entityOfLastElement = mIndexToEntityMap[ indexOfLastElement ];
		mEntityToIndexMap[ entityOfLastElement ] = indexOfRemovedEntity;// �ŏI�v�fEntity -> �󂢂��v�f�ԍ�
		mIndexToEntityMap[ indexOfRemovedEntity ] = entityOfLastElement;// �󂢂��v�f�ԍ� -> �ŏI�v�fEntity

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

	// Entity���g�p����R���|�[�l���g�f�[�^���i�[���郁����. ���Ԃ��炯�ɂȂ��Ă�\������
	OreOreLib::StaticArray<T, MAX_ENTITIES>	mComponentArray;

	// �G���e�B�e�BID����mComponentArray�C���f�b�N�X�ւ̕ϊ��e�[�u��
	OreOreLib::HashMap<Entity, size_t, 128>	mEntityToIndexMap;

	// �ʂ��ԍ��ŗ��p���̗v�f���������邽�߂̃C���f�b�N�X�e�[�u��
	OreOreLib::HashMap<Entity, size_t, 128>	mIndexToEntityMap;

	size_t	mSize;
};


#endif // !COMPONENT_ARRAY_H
