#ifndef RANDOMGLOBALS_H
#define RANDOMGLOBALS_H
#include <random>


class RandomGlobals {
public:
    // Access the single shared instance:
    static RandomGlobals& instance() {
        static RandomGlobals inst;
        return inst;
    }

    // Reseed (call once at startup):
    void seed(unsigned s) { _rng.seed(s); }

    // Uniform [0,1):
    double rand01() {
        std::uniform_real_distribution<double> d(0.0, 1.0);
        return d(_rng);
    }

    // Uniform integer in [min,max]:
    int randInt(int min, int max) {
        std::uniform_int_distribution<int> d(min, max);
        return d(_rng);
    }

private:
    RandomGlobals()
      : _rng(std::random_device{}()) {}      // seeded at first use
    std::mt19937 _rng;
};


#endif //RANDOMGLOBALS_H
