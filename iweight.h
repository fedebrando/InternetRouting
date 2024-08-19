using namespace std;

class IWeight
{
    public:
        IWeight() {};
        //virtual IWeight operator + (const IWeight& other) const = 0;
        //virtual IWeight operator * (const IWeight& other) const = 0;
        virtual IWeight zero(void) const = 0;
        //virtual IWeight unity(void) const = 0;
        virtual ~IWeight() {};
};
