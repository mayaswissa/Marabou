// #ifndef DQNRUNNER_H
// #define DQNRUNNER_H
//
// #include "DQNAgent.h"
// #include "DQNEnvironment.h"
// #include <deque>
// #include <vector>
//
// class RunDQN
// {
// public:
//     RunDQN(Environment &env, Agent &agent, unsigned nEpisodes = 2000);
//
//     /**
//      * Simulate a single episode of interaction in the environment using the current policy
//      */
//     void run() const;
//
//     /**
//      * Train the agent over multiple episodes and adjusts the agent's policy.
//      */
//     void dqnLearn(unsigned max_t, float eps_start, float eps_end, float eps_decay) const;
//
//
// private:
//     Agent& _agent;
//     Environment& _environment;
//     std::vector<float> scores;
//     std::deque<float> _scoresWindow;
//     unsigned _nEpisodes;
// };
// #endif