
#include <concepts>
#include <iostream>

using namespace std;

template<typename T>
concept Weighable = requires(T a, T b)
{
    { a + b } -> convertible_to<T>;
    { a * b } -> convertible_to<T>;
    { T::zero() } -> convertible_to<T>;
    { T::unity() } -> convertible_to<T>;
};

class Distance
{
    private:
        double d;

    public:
        static Distance zero()
        {
            return Distance(0);
        }

        static Distance unity()
        {
            return Distance(1);
        }

        Distance(double d)
        {
            this->d = d;
        }

        Distance operator + (const Distance& other)
        {
            return Distance(d + other.d);
        }

        Distance operator * (const Distance& other)
        {
            return Distance(d * other.d);
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