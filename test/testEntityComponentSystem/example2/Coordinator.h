#ifndef COORDINATOR_H
#define	COORDINATOR_H

#include	"ComponentManager.h"
#include	"EntityManager.h"
#include	"SystemManager.h"



class Coordinator
{
public:

	void Init()
	{

	}


	//================ Entity methods ===============//
	// エンティティ(ID)を作成する
	Entity CreateEntity()
	{
		return mEntityManager.CreateEntity();
	}

	// エンティティを解放する
	void DestroyEntity( Entity entity )
	{
		mEntityManager.DestroyEntity( entity );
		mComponentManager.EntityDestroyed( entity );
		mSystemManager.EntityDestroyed( entity );
	}


	//=============== Component methods ===============//
	// 型を指定してコンポーネントを作成する
	template < typename T >
	void RegisterComponent()
	{
		mComponentManager.RegisterComponent<T>();
	}


	// エンティティにコンポーネントを追加する
	template < typename T >
	void AddComponent( Entity entity, T component )
	{
		// entityが使うコンポーネントデータ領域を確保する
		mComponentManager.AddComponent<T>( entity, component );

		// entityに対応するSignatureを更新する
		auto signature = mEntityManager.GetSignature( entity );
		signature.Set( mComponentManager.GetComponentType<T>() );// コンポーネントIDをSignatureに追記する
		mEntityManager.SetSignature( entity, signature );

		// Signature更新をシステムに反映する
		mSystemManager.EntitySignatureChanged( entity, signature );
	}


	template < typename T >
	void RemoveComponent( Entity entity )
	{
		mComponentManager.RemoveComponent<T>( entity );

		auto signature = mEntityManager.GetSignature( entity );
		signature.Unset( mComponentManager.GetComponentType<T>() );
		mEntityManager.SetSignature( entity, signature );

		mSystemManager.EntitySignatureChanged( entity, signature );
	}

	
	template < typename T >
	T& GetComponent( Entity entity )
	{
		return mComponentManager.GetComponent<T>( entity );
	}


	template < typename T >
	ComponentType GetComponentType()
	{
		return mComponentManager.GetComponentType<T>();
	}


	//================= System methods =================//
	template < typename T >
	T* RegisterSystem()
	{
		return mSystemManager.RegisterSystem<T>();
	}


	template < typename T >
	void SetSystemSignature( Signature signature )
	{
		mSystemManager.SetSignature<T>( signature );
	}



private:

	ComponentManager	mComponentManager;
	EntityManager		mEntityManager;
	SystemManager		mSystemManager;
};



#endif // !COORDINATOR_H
