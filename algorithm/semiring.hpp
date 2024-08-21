
/*
 * Math semiring metric
*/

#ifndef SEMIRING
#define SEMIRING

#include <concepts>

using namespace std;

// Abstract class Weighable to allow classes defining their metric
template<typename E>
class Weighable
{
    public:
        // Associative and commutative operator to select best path(s)
        virtual E operator + (const E& other) const = 0;

        // Associative operator to measure paths
        virtual E operator * (const E& other) const = 0;

        virtual bool operator == (const E& other) const = 0;

        virtual bool operator != (const E& other) const final
        {
            return !(*this == other);
        }

        // At-least-total order relation defined from + operator
        virtual bool operator <= (const E& other) const final
        {
            return *this == *this + other;
        }

        // At-least-total strong order relation defined from + and == operators
        virtual bool operator < (const E& other) const final 
        {
            return *this <= other && *this != other;
        }
};

template<typename E>
concept Semiring = requires()
{
    // The identity for + (a + 0 = 0 + a = a), and the annihilator for * (a * 0 = 0 * a = 0)
    { E::zero() } -> convertible_to<E>;

    // The identity for * (a * 1 = 1 * a = a)
    { E::unity() } -> convertible_to<E>;
    
    is_base_of_v<Weighable<E>, E>;
};

#endif