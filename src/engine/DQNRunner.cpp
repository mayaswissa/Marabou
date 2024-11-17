// #include "DQNRunner.h"
//
// RunDQN::RunDQN(Environment &env, Agent& agent, const unsigned nEpisodes)
//     : _agent(agent), _environment(env), _scoresWindow(100), _nEpisodes(nEpisodes){}
//
// void RunDQN::run() const {
//     torch::Tensor state, nextState;
//     unsigned reward;
//     bool done;
//     unsigned totalReward = 0.0;
//     _environment.reset(state);
//     while (!done) {
//         auto action = _agent.act(state, 0.01);
//         _environment.step(action.actionToTensor(), nextState, reward, done);
//         _agent.step(state, action.actionToTensor(), reward, nextState, done);
//         totalReward += reward;
//         state = nextState;
//     }
// }
//
//
// void RunDQN::dqnLearn(const unsigned max_t, const float eps_start, const float eps_end, const float eps_decay) const {
//     float eps = eps_start;
//     std::vector<float> scores;
//     std::deque<float> scores_window;
//     torch::Tensor state, nextState;
//     unsigned reward;
//     bool done;
//     for (unsigned int episode = 1; episode <= _nEpisodes; ++episode) {
//         _environment.reset(state);
//         float score = 0;
//
//         for (unsigned t = 0; t < max_t; ++t) {
//             const auto action = _agent.act(state, eps);
//             _environment.step(action.actionToTensor(), nextState, reward, done);
//             score += reward;
//             _agent.step(state, action.actionToTensor(), reward, nextState, done);
//             state = nextState;
//             if (done) break;
//         }
//
//         // Manage score tracking and reporting
//         scores.push_back(score);
//         scores_window.push_back(score);
//         if (scores_window.size() > 100) scores_window.pop_front();
//
//         // Epsilon decay
//         eps = std::max(eps_end, eps * eps_decay);
//
//         if (episode % 100 == 0) {
//             const float average_score = std::accumulate(scores_window.begin(), scores_window.end(), 0.0) / scores_window.size();
//             std::cout << "Episode " << episode << "\tAverage Score: " << average_score << std::endl;
//         }
//
//     }
// }
//
//
