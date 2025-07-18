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
#include "RandomGlobals.h"

#include <cstdlib>
#include <dirent.h>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <utility>
#include <vector>

#ifdef ENABLE_OPENBLAS
#include "cblas.h"
#endif

#define DQN_LOG( x, ... ) MARABOU_LOG( GlobalConfiguration::DQN_LOGGING, "DQN: %s\n", x )


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

std::vector<std::string> getEpsFiles( const std::string &examplePath )
{
    size_t pos = examplePath.find_last_of( '/' );
    if ( pos == std::string::npos )
    {
        std::cerr << "Invalid path format." << std::endl;
        exit( 1 );
    }

    std::string parentFolder = examplePath.substr( 0, pos );

    DIR *dir = opendir( parentFolder.c_str() );
    if ( dir == nullptr )
    {
        perror( "opendir failed" );
        exit( 1 );
    }

    struct dirent *entry;
    std::vector<std::string> files;

    while ( ( entry = readdir( dir ) ) != nullptr )
    {
        std::string filename( entry->d_name );
        if ( filename == "." || filename == ".." )
            continue;
        if ( filename.find( "eps" ) != std::string::npos &&
             filename.find( ".txt" ) != std::string::npos )
        {
            files.push_back( filename );
        }
    }
    closedir( dir );

    std::sort( files.begin(), files.end() );

    return files;
}

void extractNetworkName( std::string &network )
{
    String networkFilePath = Options::get()->getString( Options::INPUT_FILE_PATH );
    std::string networkPath = static_cast<std::string>( networkFilePath.ascii() );
    size_t start = networkPath.find_last_of( '/' );
    size_t end = networkFilePath.find( ".onnx" );
    network = networkPath.substr( start + 1, end - start - 1 );
}

void extractTrainedAgentID( std::string &trainedAgentPath, std::string &trainedAgentID )
{
    trainedAgentPath = Options::get()->getString( Options::DQN_AGENT_NETWORKS_PATH ).ascii();
    size_t start = trainedAgentPath.find_last_of( '/' );
    size_t end = trainedAgentPath.find_last_of( '_' );
    auto t = trainedAgentPath.substr( start + 1, end - ( start + 1 ) );
    trainedAgentID = t;
}

void extractExampleID( std::string &examplePath, std::string &exampleID )
{
    examplePath = Options::get()->getString( Options::PROPERTY_FILE_PATH ).ascii();
    size_t ex_pos = examplePath.find( "ex_" ) + 3;
    size_t label_pos = examplePath.find( "_label_" ) + 7;
    std::string ex_id = examplePath.substr( ex_pos, 4 );
    std::string label_id = examplePath.substr( label_pos, 1 );
    exampleID = ex_id + label_id;
}

void extractMetaroomID( std::string &examplePath, std::string &exampleID )
{
    examplePath = Options::get()->getString( Options::PROPERTY_FILE_PATH ).ascii();
    size_t idx_start = examplePath.find( "spec_idx_" );
    size_t eps_start = examplePath.find( "_eps_" );
    size_t dot_pos = examplePath.find( ".vnnlib" );
    if ( idx_start == std::string::npos || eps_start == std::string::npos ||
         dot_pos == std::string::npos )
    {
        std::cerr << "Error: Unexpected file name format: " << examplePath << std::endl;
        exit( 1 );
    }
    idx_start += 9;
    size_t idx_end = eps_start;
    std::string idx = examplePath.substr( idx_start, idx_end - idx_start );
    size_t eps_val_start = eps_start + 5;
    std::string eps = examplePath.substr( eps_val_start, dot_pos - eps_val_start );
    eps.erase( std::remove( eps.begin(), eps.end(), '.' ), eps.end() );
    exampleID = idx + eps;
}

void extractCoraID( std::string &examplePath, std::string &exampleID )
{
    examplePath = Options::get()->getString( Options::PROPERTY_FILE_PATH ).ascii();
    size_t start = examplePath.find( "mnist-img" ) + strlen( "mnist-img" );
    size_t end = examplePath.find( ".vnnlib", start );
    std::string num = examplePath.substr( start, end - start );
    while ( num.size() < 3 )
        num = "0" + num;

    exampleID = num;
}


std::string parentDir( const std::string &path )
{
    auto pos = path.find_last_of( '/' );
    if ( pos == std::string::npos )
        return "";
    return path.substr( 0, pos );
}

bool isDir( const std::string &path )
{
    struct stat st;
    if ( stat( path.c_str(), &st ) != 0 )
    {
        return false;
    }
    return S_ISDIR( st.st_mode );
}

std::vector<std::string> listDir( const std::string &dirPath )
{
    std::vector<std::string> names;
    DIR *dir = opendir( dirPath.c_str() );
    if ( !dir )
    {
        std::cerr << "opendir failed on \"" << dirPath << "\": " << strerror( errno ) << "\n";
        return names;
    }
    struct dirent *entry;
    while ( ( entry = readdir( dir ) ) != nullptr )
    {
        std::string name = entry->d_name;
        if ( name == "." || name == ".." )
            continue;
        names.push_back( name );
    }
    closedir( dir );
    std::sort( names.begin(), names.end() );
    return names;
}

void trainAgentOnExamples( Options *options,
                           const std::vector<std::pair<std::string, std::string>> &examples,
                           std::unique_ptr<Agent> &agent,
                           int *numSplits,
                           std::ofstream &outputTxtFile )
{
    unsigned DQN_epochs = options->getInt( Options::DQN_EPOCHS );
    unsigned learnGuidedSteps = options->getInt( Options::DQN_GUIDED_STEPS ); // pretrain steps
    double epsilon = GlobalConfiguration::DQN_EPSILON_START;
    agent = nullptr;
    if ( !outputTxtFile.is_open() )
        return;
    outputTxtFile << "\n    results of each episode : \n" << std::flush;

    // 1) COLLECT DEMONSTRATION TRAJECTORIES
    DQN_LOG( "=== COLLECTING DEMOS ===\n" );
    GlobalConfiguration::DON_TRAINING_PHASE = 0;
    for ( auto &ex : examples )
    {
        options->setString( Options::PROPERTY_FILE_PATH, ex.first );
        int splits = 0;
        GlobalConfiguration::DQN_FORCED_HEURISTIC = GlobalConfiguration::GuidedHeuristic::POLARITY;
        agent = Marabou().trainDQNAgent( epsilon, ex.second, std::move( agent ), &splits );
        *numSplits += splits;
        GlobalConfiguration::DQN_FORCED_HEURISTIC = GlobalConfiguration::GuidedHeuristic::BABS_R;
        splits = 0;
        agent = Marabou().trainDQNAgent( epsilon, ex.second, std::move( agent ), &splits );
        *numSplits += splits;
    }

    // 2) PRE‐TRAIN ON THE DEMOS
    DQN_LOG( "=== PRE‐TRAINING ON DEMOS ===\n" );
    GlobalConfiguration::DON_TRAINING_PHASE = 1;
    for ( unsigned i = 0; i < learnGuidedSteps; ++i )
        agent->learn();

    // 3) ONLINE RL
    DQN_LOG( "=== ONLINE RL PHASE ===\n" );
    GlobalConfiguration::DON_TRAINING_PHASE = 2;
    for ( unsigned epoch = 0; epoch < DQN_epochs; ++epoch )
    {
        for ( auto &[path, id] : examples )
        {
            options->setString( Options::PROPERTY_FILE_PATH, path );
            int splits = 0;
            agent = Marabou().trainDQNAgent( epsilon, id, std::move( agent ), &splits );
            *numSplits += splits;
        }
        epsilon = std::max( GlobalConfiguration::DQN_EPSILON_END,
                            epsilon * GlobalConfiguration::DQN_EPSILON_DECAY );
        // agent->schedulersStep();
        outputTxtFile << Stringf( "Completed RL epoch %u, epilon=%.4f\n", epoch, epsilon ).ascii()
                      << std::flush;
    }

    if ( agent != nullptr &&
         *numSplits > static_cast<int>( Options::get()->getInt( Options::DQN_BATCH_SIZE ) * 20 ) )
    {
        const auto path = options->getString( Options::DQN_AGENT_NETWORKS_PATH );
        const std::string filePath = std::string( path.ascii() ) + "/agent";
        agent->saveNetworks( filePath );
        outputTxtFile << "agent network has been saved. Path: " << filePath;
        outputTxtFile << std::flush;
    }

    outputTxtFile << "\n";
}


void setRandomSeed()
{
    // unsigned seedVal = static_cast<unsigned>( std::time( nullptr ) );
    RandomGlobals::instance().seed( 1 );
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
            auto const mode = options->getInt( Options::DQN_MODE );
            setRandomSeed();
            std::ostringstream currentRunFile;
            std::string examplePath;
            std::string exampleID;
            auto const exampleType = options->getString( Options::BENCHMARK );
            if ( exampleType == "metaroom" )
                extractMetaroomID( examplePath, exampleID );
            else if ( exampleType == "cora" )
                extractCoraID( examplePath, exampleID );
            else
                extractExampleID( examplePath, exampleID );
            auto txtOutputFilePath = options->getString( Options::DQN_OUTPUT_FILE_PATH );
            std::string network;
            extractNetworkName( network );
            currentRunFile << std::string( txtOutputFilePath.ascii() ) << exampleID << ".txt";
            options->setString( Options::SUMMARY_FILE, currentRunFile.str() );
            std::ofstream outFile( currentRunFile.str(), std::ios::out | std::ios::app );
            outFile << "Example: " << exampleID << ". Network : " << network << "\n";
            outFile.flush();
            if ( !outFile )
            {
                std::cerr << "Failed to open " << currentRunFile.str() << "\n";
                return 1;
            }
            if ( mode == 1 )
            {
                // train
                std::string root = parentDir( examplePath );
                if ( root.empty() || !isDir( root ) )
                {
                    std::cerr << "Error: cannot determine root from '" << examplePath << "'\n";
                    return 1;
                }
                std::vector<std::pair<std::string, std::string>> examples;
                for ( auto &fname : listDir( root ) )
                {
                    if ( fname.size() < 4 || fname.substr( fname.size() - 4 ) != ".txt" )
                        continue;
                    std::string fullPath = root + "/" + fname;
                    std::string currentID;
                    if ( exampleType == "metaroom" )
                        extractMetaroomID( fullPath, currentID );
                    else if ( exampleType == "cora" )
                        extractCoraID( fullPath, currentID );
                    else
                        extractExampleID( fullPath, currentID );
                    examples.emplace_back( fullPath, currentID );
                }

                // -------- now randomly sample only M of them --------
                int M = options->getInt( Options::DQN_N_EXAMPLES );
                if ( (int)examples.size() > M )
                {
                    std::unordered_set<size_t> picks;
                    while ( picks.size() < (size_t)M )
                    {
                        picks.insert( RandomGlobals::instance().randInt( 0, examples.size() - 1 ) );
                    }
                    std::vector<std::pair<std::string, std::string>> sampled;
                    sampled.reserve( M );
                    for ( auto idx : picks )
                        sampled.push_back( examples[idx] );
                    examples.swap( sampled );
                }
                options->setString( Options::SPLITTING_STRATEGY, "DQN-agent" );
                struct timespec startTraining = TimeUtils::sampleMicro();
                int numSplits = 0;
                outFile << std::flush;
                DQN_LOG(
                    Stringf( "Start training agent on example: %s  ", exampleID.c_str() ).ascii() );
                std::unique_ptr<Agent> agent;
                trainAgentOnExamples( options, examples, agent, &numSplits, outFile );

                struct timespec endTraining = TimeUtils::sampleMicro();
                unsigned long long totalTraining =
                    TimeUtils::timePassed( startTraining, endTraining );
                outFile << "\t Done training"
                        << ". Time : " << totalTraining << " Splits : " << numSplits << "\n";
            }
            else if ( mode == 2 )
            {
                // run
                options->setString( Options::SPLITTING_STRATEGY, "DQN-agent" );
                std::string agentPath =
                    options->getString( Options::DQN_AGENT_NETWORKS_PATH ).ascii();
                if ( !std::ifstream( agentPath + "_local.pth" ) )
                {
                    std::cout << "trained agent path does not exist.\n";
                    return 0;
                }
                std::string trainedAgentPath;
                std::string trainedAgentID;
                extractTrainedAgentID( trainedAgentPath, trainedAgentID );
                outFile << "Trained on agent : " << trainedAgentID << "\n";
                outFile.flush();
                std::string root = parentDir( examplePath );
                if ( root.empty() || !isDir( root ) )
                {
                    std::cerr << "Error: cannot determine root from '" << root << "'\n";
                    return 1;
                }
                if ( exampleType == "property1" )
                {
                    auto examples = listDir( root );
                    for ( auto &currentExample : examples )
                    {
                        if ( currentExample.size() < 4 ||
                             currentExample.substr( currentExample.size() - 4 ) != ".txt" )
                            continue;
                        std::string fullCurrentExamplePath = root + "/" + currentExample;
                        size_t ex_pos = fullCurrentExamplePath.find( "ex_" ) + 3;
                        size_t label_pos = fullCurrentExamplePath.find( "_label_" ) + 7;
                        size_t eps_pos = fullCurrentExamplePath.find( "eps" ) + 3;
                        std::string ex_id = fullCurrentExamplePath.substr( ex_pos, 4 );
                        std::string label_id = fullCurrentExamplePath.substr( label_pos, 1 );
                        std::string eps_id = fullCurrentExamplePath.substr( eps_pos, 2 );
                        std::string currentExampleID = ex_id + label_id + eps_id;
                        currentExampleID += eps_id;
                        options->setString( Options::PROPERTY_FILE_PATH, fullCurrentExamplePath );
                        struct timespec startTime = TimeUtils::sampleMicro();
                        int numSplits = 0;
                        outFile << "epsilon : " << eps_id << "\n";
                        outFile << std::flush;
                        DQN_LOG( Stringf( "Start runing trained agent with example: %s  ",
                                          currentExampleID.c_str() )
                                     .ascii() );
                        Marabou().runTrainedAgentOnExample( &numSplits );
                        struct timespec endTime = TimeUtils::sampleMicro();
                        unsigned long long totalTraining =
                            TimeUtils::timePassed( startTime, endTime );

                        DQN_LOG(
                            Stringf( "Done solving. Time : %llu milli. \n", totalTraining / 1000 )
                                .ascii() );
                    }
                }
                else
                {
                    int numSplits = 0;
                    Marabou().runTrainedAgentOnExample( &numSplits );
                }
            }
            else
            {
                auto spittingHeuristic = options->getString( Options::SPLITTING_STRATEGY );
                outFile << "Strategy : " << std::string( spittingHeuristic.ascii() ) << "\n";
                outFile.flush();
                if ( exampleType == "property1" )
                {
                    std::string root = parentDir( examplePath );
                    if ( root.empty() || !isDir( root ) )
                    {
                        std::cerr << "Error: cannot determine root from '" << root << "'\n";
                        return 1;
                    }
                    auto examples = listDir( root );
                    for ( auto &currentExample : examples )
                    {
                        if ( currentExample.size() < 4 ||
                             currentExample.substr( currentExample.size() - 4 ) != ".txt" )
                            continue;
                        std::string fullCurrentExamplePath = root + "/" + currentExample;
                        size_t ex_pos = fullCurrentExamplePath.find( "ex_" ) + 3;
                        size_t label_pos = fullCurrentExamplePath.find( "_label_" ) + 7;
                        size_t eps_pos = fullCurrentExamplePath.find( "eps" ) + 3;
                        std::string ex_id = fullCurrentExamplePath.substr( ex_pos, 4 );
                        std::string label_id = fullCurrentExamplePath.substr( label_pos, 1 );
                        std::string eps_id = fullCurrentExamplePath.substr( eps_pos, 2 );
                        std::string currentExampleID = ex_id + label_id + eps_id;
                        options->setString( Options::PROPERTY_FILE_PATH, fullCurrentExamplePath );
                        struct timespec startTime = TimeUtils::sampleMicro();
                        outFile << "epsilon : " << eps_id << "\n";
                        outFile << std::flush;
                        Marabou().run();
                        struct timespec endTime = TimeUtils::sampleMicro();
                        unsigned long long totalTraining =
                            TimeUtils::timePassed( startTime, endTime );
                        DQN_LOG(
                            Stringf( "Done solving. Time : %llu milli. \n", totalTraining / 1000 )
                                .ascii() );
                    }
                }
                else
                {
                    Marabou().run();
                }
            }
            outFile.close();
            return 0;
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
