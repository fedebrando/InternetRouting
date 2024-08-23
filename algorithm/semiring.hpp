
/*
 * Math semiring metric
*/

#ifndef SEMIRING
#define SEMIRING

#include <concepts>
#include <utility>
#include "utilities.hpp"

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

// The lexicographic product of two semirings is
// distributive (and optimality is garanteed) if and only if 
// the two components are distributive
// and the first one * is cancellative or the second one * is constant
template <Semiring E1, Semiring E2>
requires Print<E1> && Print<E2>
class LexProduct : public Weighable<LexProduct<E1, E2>>
{
    private:
        pair<E1, E2> couple;

    public:
        static LexProduct<E1, E2> zero()
        {
            static LexProduct<E1, E2> lp0(E1::zero(), E2::zero());

            return lp0;
        }

        static LexProduct<E1, E2> unity()
        {
            static LexProduct<E1, E2> lp1(E1::unity(), E2::unity());

            return lp1;
        }

        LexProduct() : LexProduct(LexProduct::zero())
        {}

        LexProduct(const LexProduct& lp) : LexProduct(lp.getFirst(), lp.getSecond())
        {}

        LexProduct(E1 w1, E2 w2)
        {
            setFirst(w1);
            setSecond(w2);
        }

        E1 getFirst() const
        {
            return couple.first;
        }

        E2 getSecond() const
        {
            return couple.second;
        }

        void setFirst(E1 fst) 
        {
            couple.first = fst;
        }

        void setSecond(E2 snd) 
        {
            couple.second = snd;
        }

        LexProduct operator + (const LexProduct& other) const
        {
            return (couple < other.couple) ? *this : other;
        }

        LexProduct operator * (const LexProduct& other) const
        {
            return LexProduct(getFirst() * other.getFirst(), getSecond() * other.getSecond());
        }

        bool operator == (const LexProduct& other) const
        {
            return couple == other.couple;
        }

        friend ostream& operator << (ostream& os, const LexProduct& lp)
        {
            os << "(" << lp.getFirst() << ", " << lp.getSecond() << ")";
            return os;
        }

        ~LexProduct() = default;
};

#endif