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
                unsigned epochs = 500;
                double currEpisodeScore = 0;
                std::vector<double> learningRates = { 1e-2 };

                std::vector<double> alphas = { 0, 0.2, 0.4, 0.8, 1 };
                std::vector<unsigned> batchSizes = {8, 16, 32 };
                std::vector<unsigned> bufferSizes = {  64, 124, 256, 512 };
                int numRuns = 12;
                unsigned bestBufferSize = 64;
                unsigned bestBatchSize = 64;
                double bestLR = 0;
                double bestAlpha = 0;
                double minNumSplits = 30000;

                for ( auto bufferSize : bufferSizes )
                {
                    for ( auto batchSize : batchSizes )
                    {
                        for ( auto lr : learningRates )
                        {
                            for ( auto alpha : alphas )
                            {
                                int avgNumSplits = 0;
                                int numSplits = 0;
                                GlobalConfiguration::DQN_BUFFER_SIZE = bufferSize;
                                GlobalConfiguration::DQN_BATCH_SIZE = batchSize;
                                GlobalConfiguration::DQN_ALPHA_REWARDS = alpha;
                                GlobalConfiguration::DQN_LR = lr;
                                printf("learning rate = %g\n", GlobalConfiguration::DQN_LR);
                                printf("alpha = %g\n", GlobalConfiguration::DQN_ALPHA_REWARDS);
                                printf("bufferSize = %u\n", GlobalConfiguration::DQN_BUFFER_SIZE);
                                printf("batchSize = %u\n", GlobalConfiguration::DQN_BATCH_SIZE);
                                std::ostringstream currentRunFile;
                                currentRunFile
                                    << "/home/maya-swisa/Documents/Lab/DRL/Marabou/schedulerResults/"
                                    << "buffer-" << bufferSize << "batchSize-" << batchSize
                                    << "results_lr-" << std::scientific << std::setprecision( 1 )
                                    << lr << "_alpha-" << std::fixed << std::setprecision( 2 )
                                    << alpha << ".txt";

                                // Open file with generated name
                                std::ofstream outFile( currentRunFile.str() );

                                if ( outFile.is_open() )
                                {
                                    outFile << "Results for buffer size: " << bufferSize
                                            << "batchSize" << batchSize << ", learning rate: " << lr
                                            << " and alpha: " << alpha << "\n";
                                    for ( int i = 0; i < numRuns; i++ )
                                    {
                                        numSplits = 0;
                                        // GlobalConfiguration::USE_DEEPSOI_LOCAL_SEARCH = false;
                                        GlobalConfiguration::USE_DQN = true;
                                        std::unique_ptr<Agent> agent = nullptr;
                                        double epsilon = GlobalConfiguration::DQN_EPSILON_START;
                                        for ( unsigned int episode = 0; episode < epochs;
                                              ++episode )
                                        {
                                            currEpisodeScore = 0;
                                            agent = Marabou().runAgentTraining(
                                                epsilon, true, std::move( agent ) );
                                            printf( "done one train, score: %f\n",
                                                    currEpisodeScore );
                                            fflush( stdout );
                                            epsilon = std::max(
                                                GlobalConfiguration::DQN_EPSILON_END,
                                                epsilon * GlobalConfiguration::DQN_EPSILON_DECAY );
                                            // if (agent)
                                            //     agent->schedulersStep();
                                        }
                                        printf( "start solving with trained agent\n" );
                                        fflush( stdout );
                                        GlobalConfiguration::USE_DQN = true;
                                        GlobalConfiguration::USE_DEEPSOI_LOCAL_SEARCH = true;
                                        if ( agent != nullptr )
                                            agent->saveNetworks();
                                        Marabou().runAgentTraining(
                                            GlobalConfiguration::DQN_EPSILON_END, false, std::move( agent ), &numSplits );
                                        avgNumSplits += numSplits;
                                        printf( "numsplits marabouMain: %d\n", numSplits );
                                        fflush( stdout );
                                        outFile << numSplits << " ";
                                        outFile << std::flush;

                                    }
                                    avgNumSplits /= numRuns;
                                    if ( minNumSplits > avgNumSplits )
                                    {
                                        minNumSplits = avgNumSplits;
                                        bestBufferSize = bufferSize;
                                        bestBatchSize = batchSize;
                                        bestLR = lr;
                                        bestAlpha = alpha;
                                    }
                                    outFile << "\n number of splits for BufferSize " << bufferSize
                                            << "BatchSize : " << batchSize << " learning rate "
                                            << lr << " and alpha " << alpha << " :" << avgNumSplits
                                            << "\n";
                                    outFile.close();
                                }
                                else
                                {
                                    std::cerr << "Failed to open file: " << currentRunFile.str()
                                              << std::endl;
                                }
                            }
                        }
                    }
                }

                std::ostringstream bestRunFile;
                bestRunFile << "/home/maya-swisa/Documents/Lab/DRL/Marabou/schedulerResults/"
                            << "bestCombination.txt";
                std::ofstream outFile( bestRunFile.str() );
                if ( outFile.is_open() )
                {
                    outFile << "\n best combination BufferSize: "<< bestBufferSize << ", Batch Size: " << bestBatchSize
                            << ", RL:" << bestLR << " and alpha: " << bestAlpha
                            << ". Avg number of splits:" << minNumSplits << "\n";
                    outFile.close();
                }
                else
                {
                    std::cerr << "Failed to open file: " << bestRunFile.str() << std::endl;
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
