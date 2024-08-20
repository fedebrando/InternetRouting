
#ifndef SEMIRING
#define SEMIRING

#include <concepts>

using namespace std;

template<typename T>
class Weighable
{
    public:
        virtual T operator + (const T& other) const = 0;
        virtual T operator * (const T& other) const = 0;
        virtual bool operator < (const T& other) const final
        {
            return *this == *this + other;
        }
        virtual bool operator == (const T& other) const = 0;
        virtual bool operator <= (const T& other) const final 
        {
            return *this < other || *this == other;
        }
};

template<typename T>
concept Semiring = requires(T a, T b)
{
    { T::zero() } -> convertible_to<T>;
    { T::unity() } -> convertible_to<T>;
    is_base_of_v<Weighable<T>, T>;
};

#endif