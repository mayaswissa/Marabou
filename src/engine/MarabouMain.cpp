/*********************                                                        */
/*! \file MarabouMain.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz, Haoze Wu
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

 **/

#include "ConfigurationError.h"
#include "DnCMarabou.h"
#include "Error.h"
#include "LPSolverType.h"
#include "Marabou.h"
#include "Options.h"
#include <fstream>

#ifdef ENABLE_OPENBLAS
#include "cblas.h"
#endif

static std::string getCompiler()
{
    std::stringstream ss;
#ifdef __GNUC__
    ss << "GCC";
#else  /* __GNUC__ */
    ss << "unknown compiler";
#endif /* __GNUC__ */
#ifdef __VERSION__
    ss << " version " << __VERSION__;
#else  /* __VERSION__ */
    ss << ", unknown version";
#endif /* __VERSION__ */
    return ss.str();
}

static std::string getCompiledDateTime()
{
    return __DATE__ " " __TIME__;
}

void printVersion()
{
    std::cout << "Marabou version " << MARABOU_VERSION << " [" << GIT_BRANCH << " "
              << GIT_COMMIT_HASH << "]"
              << "\ncompiled with " << getCompiler() << "\non " << getCompiledDateTime()
              << std::endl;
}

void printHelpMessage()
{
    printVersion();
    Options::get()->printHelpMessage();
}

int marabouMain( int argc, char **argv )
{
    try
    {
        Options *options = Options::get();
        options->parseOptions( argc, argv );

        if ( options->getBool( Options::HELP ) )
        {
            printHelpMessage();
            return 0;
        };

        if ( options->getBool( Options::VERSION ) )
        {
            printVersion();
            return 0;
        };

        if ( options->getBool( Options::PRODUCE_PROOFS ) )
        {
            GlobalConfiguration::USE_DEEPSOI_LOCAL_SEARCH = false;
            printf( "Proof production is not yet supported with DEEPSOI search, turning search "
                    "off.\n" );
        }

        if ( options->getBool( Options::PRODUCE_PROOFS ) &&
             ( options->getBool( Options::DNC_MODE ) ) )
        {
            options->setBool( Options::DNC_MODE, false );
            printf( "Proof production is not yet supported with snc mode, turning --snc off.\n" );
        }

        if ( options->getBool( Options::PRODUCE_PROOFS ) &&
             ( options->getBool( Options::SOLVE_WITH_MILP ) ) )
        {
            options->setBool( Options::SOLVE_WITH_MILP, false );
            printf(
                "Proof production is not yet supported with MILP solvers, turning --milp off.\n" );
        }

        if ( options->getBool( Options::PRODUCE_PROOFS ) &&
             ( options->getLPSolverType() == LPSolverType::GUROBI ) )
        {
            options->setString( Options::LP_SOLVER, "native" );
            printf( "Proof production is not yet supported with MILP solvers, using native simplex "
                    "engine.\n" );
        }

        if ( options->getBool( Options::DNC_MODE ) &&
             options->getBool( Options::PARALLEL_DEEPSOI ) )
        {
            throw ConfigurationError( ConfigurationError::INCOMPTATIBLE_OPTIONS,
                                      "Cannot set both --snc and --poi to true..." );
        }

        if ( options->getBool( Options::PARALLEL_DEEPSOI ) &&
             ( options->getBool( Options::SOLVE_WITH_MILP ) ) )
        {
            options->setBool( Options::SOLVE_WITH_MILP, false );
            printf( "Cannot set both --poi and --milp to true, turning --milp off.\n" );
        }

        if ( options->getBool( Options::DNC_MODE ) ||
             ( options->getBool( Options::PARALLEL_DEEPSOI ) &&
               options->getInt( Options::NUM_WORKERS ) > 1 ) )
            DnCMarabou().run();
        else
        {
#ifdef ENABLE_OPENBLAS
            openblas_set_num_threads( options->getInt( Options::NUM_BLAS_THREADS ) );
#endif
            if ( GlobalConfiguration::USE_DQN )
            {
                unsigned _nEpisodes = 10;
                double currEpisodeScore = 0;
                std::unique_ptr<Agent> agent = nullptr;
                double epsilon = GlobalConfiguration::DQN_EPSILON_START;
                std::vector<double> learningRates = {
                    1e-5, 5e-5,  // Very small learning rates
                    1e-4, 5e-4,  // Small learning rates
                    1e-3, 5e-3,  // Moderate learning rates
                    1e-2,  5e-2,  // Large learning rates
                    1e-1,  5e-1 // Very large learning rates
                };

                std::vector<double> alphas = {
                    0,
                    0.05, 0.1,
                    0.15, 0.2,
                    0.25, 0.3,
                    0.35,  0.4,
                    0.45,  0.5,
                    0.55,  0.6,
                    0.65,  0.7,
                    0.75,  0.8,
                    0.85,  0.9,
                    0.95,  1.0,
                };
                for (auto lr: learningRates)
                {
                    int avgNumSplits = 0;
                    int numSplits = 0;
                    int numRuns = 5;
                    for (auto alpha: alphas)
                    {

                        GlobalConfiguration::DQN_ALPHA_REWARDS = alpha;
                        GlobalConfiguration::DQN_LR = lr;

                        std::ostringstream filename;
                        filename << "/home/maya-swisa/Documents/Lab/origin/Marabou/"
                                 << "results_lr-" << std::scientific << std::setprecision(1) << lr
                                 << "_alpha-" << std::fixed << std::setprecision(2) << alpha << ".txt";

                        // Open file with generated name
                        std::ofstream outFile(filename.str());

                        if (outFile.is_open())
                        {
                            outFile << "Logging results for learning rate: " << lr
                                    << " and alpha: " << alpha << "\n";
                            for (int i=0; i < numRuns; i++)
                            {
                                for ( unsigned int episode = 0; episode < _nEpisodes; ++episode )
                                {
                                    currEpisodeScore = 0;
                                    agent = Marabou().runAgentTraining( epsilon, true, std::move( agent ) );
                                    printf( "done one train, score: %f\n", currEpisodeScore );
                                    fflush( stdout );
                                    epsilon = std::max( GlobalConfiguration::DQN_EPSILON_END,
                                                        epsilon * GlobalConfiguration::DQN_EPSILON_DECAY );
                                }
                                printf( "start solving with trained agent\n" );
                                fflush( stdout );
                                GlobalConfiguration::USE_DQN = true;
                                GlobalConfiguration::USE_DEEPSOI_LOCAL_SEARCH = true;
                                if (agent != nullptr)
                                    agent->saveNetworks();
                                Marabou().runAgentTraining( 1, false, std::move(agent), &numSplits );
                                avgNumSplits += numSplits;
                            }
                            avgNumSplits /= numRuns;
                            outFile << "number of splits for learning rate " << lr
                                                << " and alpha " << alpha << " :"<< avgNumSplits << "\n";
                            outFile.close();
                        }
                        else
                        {
                            std::cerr << "Failed to open file: " << filename.str() << std::endl;
                        }
                    }
                }

                // for ( unsigned int episode = 0; episode < _nEpisodes; ++episode )
                // {
                //     currEpisodeScore = 0;
                //     agent = Marabou().runAgentTraining( epsilon, true, std::move( agent ) );
                //     printf( "done one train, score: %f\n", currEpisodeScore );
                //     fflush( stdout );
                //     epsilon = std::max( GlobalConfiguration::DQN_EPSILON_END,
                //                         epsilon * GlobalConfiguration::DQN_EPSILON_DECAY );
                // }
                //
                // // validation run:
                // GlobalConfiguration::USE_DQN = true;
                // GlobalConfiguration::DQN_TRAINING = false;
                // int numSplits = 0;
                // for ( unsigned int validations = 0; validations < 1; ++validations )
                // {
                //     printf( "Validation run\n" );
                //     fflush( stdout );
                //     currEpisodeScore = 0;
                //     agent = Marabou().runAgentTraining( epsilon, true, std::move( agent ) );
                //     epsilon = std::max( GlobalConfiguration::DQN_EPSILON_END,
                //                         epsilon * GlobalConfiguration::DQN_EPSILON_DECAY );
                // }
                // printf( "start solving with trained agent\n" );
                // fflush( stdout );
                // GlobalConfiguration::USE_DQN = true;
                // GlobalConfiguration::USE_DEEPSOI_LOCAL_SEARCH = true;
                // if (agent != nullptr)
                //     agent->saveNetworks();
                // Marabou().runAgentTraining( 1, false, std::move(agent), &numSplits );
                return 0;
            }

            Marabou().run();
        }
    }
    catch ( const Error &e )
    {
        fprintf( stderr,
                 "Caught a %s error. Code: %u, Errno: %i, Message: %s.\n",
                 e.getErrorClass(),
                 e.getCode(),
                 e.getErrno(),
                 e.getUserMessage() );

        return 1;
    }

    return 0;
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
