#ifndef PERIPHERAL_TRAITS_H
#define	PERIPHERAL_TRAITS_H


template<typename T>
T sum(T first)
{
  return first;
}


template<typename T, typename... Args>
T sum(T first, Args... args)
{
  return first + sum(args...);
}




//######################################################################//
//																		//
//									Sum									//
//																		//
//######################################################################//

template < unsigned... Ns > struct sum_;


template <>
struct sum_<>
{
   static constexpr unsigned value = 0;
};


template < unsigned Head, unsigned ... Rest >
struct sum_<Head, Rest...>
{
	static constexpr unsigned value = Head + sum_<Rest...>::value; 
};




//######################################################################//
//																		//
//								Mult									//
//																		//
//######################################################################//

template < unsigned... Ns > struct mult_;


template <>
struct mult_<>
{
   static constexpr unsigned value = 1;
};


template < unsigned Head, unsigned ... Rest >
struct mult_<Head, Rest...>
{
	static constexpr unsigned value = Head * mult_<Rest...>::value; 
};



#endif // !PERIPHERAL_TRAITS_H
