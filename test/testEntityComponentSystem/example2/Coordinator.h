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


	// Entity methods
	Entity CreateEntity()
	{
		return mEntityManager.CreateEntity();
	}


	void DestroyEntity( Entity entity )
	{
		mEntityManager.DestroyEntity( entity );
		mComponentManager.EntityDestroyed( entity );
		mSystemManager.EntityDestroyed( entity );
	}


	// Component methods
	template < typename T >
	void RegisterComponent()
	{
		mComponentManager.RegisterComponent<T>();
	}


	template < typename T >
	void AddComponent( Entity entity, T component )
	{
		mComponentManager.AddComponent<T>( entity, component );

		auto signature = mEntityManager.GetSignature( entity );
		signature.Set( mComponentManager.GetComponentType<T>() );
		mEntityManager.SetSignature( entity, signature );

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


	// System methods
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
