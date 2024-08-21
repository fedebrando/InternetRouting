
#ifndef SEMIRING
#define SEMIRING

#include <concepts>

using namespace std;

template<typename E>
class Weighable
{
    public:
        virtual E operator + (const E& other) const = 0;
        virtual E operator * (const E& other) const = 0;
        virtual bool operator == (const E& other) const = 0;

        virtual bool operator != (const E& other) const final
        {
            return !(*this == other);
        }

        virtual bool operator <= (const E& other) const final
        {
            return *this == *this + other;
        }

        virtual bool operator < (const E& other) const final 
        {
            return *this <= other && *this != other;
        }
};

template<typename E>
concept Semiring = requires()
{
    { E::zero() } -> convertible_to<E>;
    { E::unity() } -> convertible_to<E>;
    is_base_of_v<Weighable<E>, E>;
};

#endif