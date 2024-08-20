
#include <concepts>
#include <limits>
#include <iostream>

using namespace std;

template<typename T>
class Orderable
{
    public:
        virtual bool operator < (const T& other) const = 0;
        virtual bool operator == (const T& other) const = 0;
        virtual bool operator <= (const T& other) const final 
        {
            return *this < other || *this == other;
        }
};

template<typename T>
concept Weighable = requires(T a, T b)
{
    { a + b } -> convertible_to<T>;
    { a * b } -> convertible_to<T>;
    { T::zero() } -> convertible_to<T>;
    { T::unity() } -> convertible_to<T>;
    is_base_of_v<Orderable<T>, T>;
};

class Distance : public Orderable<Distance>
{
    private:
        double d;

    public:
        static Distance zero()
        {
            return Distance(numeric_limits<double>::infinity());
        }

        static Distance unity()
        {
            return Distance(0);
        }

        Distance(double d)
        {
            this->d = d;
        }

        Distance operator + (const Distance& other)
        {
            return Distance(min(d, other.d));
        }

        Distance operator * (const Distance& other)
        {
            return Distance(d + other.d);
        }

        bool operator < (const Distance& other) const override
        {
            return d == d + other.d;
        }

        bool operator == (const Distance& other) const override
        {
            return d == other.d;
        }

        friend ostream& operator << (ostream& os, const Distance& dis)
        {
            os << dis.d;
            return os;
        }
};

template<Weighable T>
T foo(T a, T b)
{
    return T::unity() + T::zero() + a * b;
}

int main(void)
{
    Distance d1(2);
    Distance d2(8);
    cout << foo(d1, d2) << endl;
    return 0;
}