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
	// �G���e�B�e�B(ID)���쐬����
	Entity CreateEntity()
	{
		return mEntityManager.CreateEntity();
	}

	// �G���e�B�e�B���������
	void DestroyEntity( Entity entity )
	{
		mEntityManager.DestroyEntity( entity );
		mComponentManager.EntityDestroyed( entity );
		mSystemManager.EntityDestroyed( entity );
	}


	//=============== Component methods ===============//
	// �^���w�肵�ăR���|�[�l���g���쐬����
	template < typename T >
	void RegisterComponent()
	{
		mComponentManager.RegisterComponent<T>();
	}


	// �G���e�B�e�B�ɃR���|�[�l���g��ǉ�����
	template < typename T >
	void AddComponent( Entity entity, T component )
	{
		// entity���g���R���|�[�l���g�f�[�^�̈���m�ۂ���
		mComponentManager.AddComponent<T>( entity, component );

		// entity�ɑΉ�����Signature���X�V����
		auto signature = mEntityManager.GetSignature( entity );
		signature.Set( mComponentManager.GetComponentType<T>() );// �R���|�[�l���gID��Signature�ɒǋL����
		mEntityManager.SetSignature( entity, signature );

		// Signature�X�V���V�X�e���ɔ��f����
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
