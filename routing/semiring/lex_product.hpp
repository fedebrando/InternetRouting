
#ifndef LEX_PRODUCT
#define LEX_PRODUCT

#include <iostream>
#include <utility>
#include "utilities.hpp"
#include "semiring.hpp"

using namespace std;

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
