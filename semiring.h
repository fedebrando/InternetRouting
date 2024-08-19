
#ifndef SEMIRING
#define SEMIRING
template<typename T>
class Semiring
{
    private:
        T (*plus)(T op1, T op2);
        T (*times)(T op1, T op2);
        T zero;
        T unity;
    public:
        Semiring(T (*plus)(T, T), T (*times)(T, T), T zero, T unity)
        {
            this->plus = plus;
            this->times = times;
            this->zero = zero;
            this->unity = unity;
        }
        ~Semiring()
        {}
        friend T operator + (const T& op1, const T& op2) const
        {
            return plus(op1, op2);
        }
        friend T operator * (const T& op1, const T& op2) const
        {
            return times(op1, op2);
        }
        T getZero() const
        {
            return zero;
        }
        T getUnity() const
        {
            return unity;
        }
};
#endif