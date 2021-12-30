#ifndef COMPONENT_MANAGER_H
#define	COMPONENT_MANAGER_H

#include	<oreore/container/HashMap.h>

#include	"ComponentArray.h"



class ComponentManager
{
public:

	template< typename T >
	void RegisterComponent()
	{
		const char* typeName = typeid(T).name();

		ASSERT( !mComponentTypes.Exists( typeName ) && "Component already exists." );

		// 型名とコンポーネントIDのペアを登録する
		mComponentTypes.Put( typeName, mNextComponentType );

		// 型名とコンポーネント配列のペアを登録する
		mComponentArrays.Put( typeName, new ComponentArray<T>() );

		// コンポーネントIDをインクリメントする
		++mNextComponentType;
	}


	template < typename T >
	ComponentType GetComponentType()
	{
		const char* typeName = typeid(T).name();

		ASSERT( mComponentTypes.Exists( typeName ) && "Component already exists." );

		return mComponentTypes[ typeName ];
	}


	template < typename T >
	void AddComponent( Entity entity, T component )
	{
		GetComponentArray<T>().InsertData( entity, component );
	}


	template < typename T >
	void RemoveComponent( Entity entity )
	{
		GetComponentArray<T>().RemoveData( entity );
	}


	template < typename T >
	T& GetComponent( Entity entity )
	{
		return GetComponentArray<T>().GetData( entity );
	}


	void EntityDestroyed( Entity entity )
	{
		for( const auto& pair : mComponentArrays )
		{
			const auto& component = pair.second;
			component->EntityDestroyed( entity );
		}
	}



private:

	// コンポーネント毎に採番したID(ComponentType)のマップ. コンポーネントの型情報をキーとしてアクセスする
	OreOreLib::HashMap<const char*, ComponentType, 128>	mComponentTypes{};

	// コンポーネント配列のマップ. コンポーネントの型情報をキーとしてアクセスする
	OreOreLib::HashMap<const char*, IComponentArray*, 128>	mComponentArrays{};

	// The component type to be assigned to the next registered component - starting at 0
	ComponentType mNextComponentType{};



	// Helper function

	template < typename T >
	ComponentArray<T>& GetComponentArray()
	{
		const char* typeName = typeid(T).name();

		ASSERT( mComponentTypes.Exists( typeName ) && "Component not registered." );

		return *static_cast< ComponentArray<T>* >( mComponentArrays[ typeName ] );
	}


};




#endif // !COMPONENT_MANAGER_H
