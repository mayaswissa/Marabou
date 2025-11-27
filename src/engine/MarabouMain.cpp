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

void extractRobustnessExampleID( std::string &examplePath, std::string &exampleID )
{
    size_t ex_pos = examplePath.find( "ex_" );
    size_t label_pos = examplePath.find( "_label_" );
    if ( ex_pos == std::string::npos || label_pos == std::string::npos )
    {
        exampleID.clear();
        return;
    }
    ex_pos += 3;
    label_pos += 7;
    std::string ex_id = examplePath.substr( ex_pos, 4 );
    std::string label_id = examplePath.substr( label_pos, 1 );
    exampleID = ex_id + label_id;
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

static bool isRobustnessRoot( const std::string &root )
{
    if ( root.empty() || !isDir( root ) )
        return false;
    std::vector<std::string> names = listDir( root );
    for ( size_t i = 0; i < names.size(); ++i )
    {
        const std::string &name = names[i];
        if ( name.find( "label" ) != std::string::npos )
            return true;
    }
    return false;
}

void extractID( std::string &path, std::string &outID )
{
    if ( path.find( "ex_" ) != std::string::npos && path.find( "_label_" ) != std::string::npos )
    {
        extractRobustnessExampleID( path, outID );
        return;
    }
    auto pos = path.find_last_of( '/' );
    if ( pos == std::string::npos )
        outID = "noID";
    else
        outID = path.substr( pos + 1 );
}

std::string computeRootDir( const std::string &examplePath )
{
    std::string root = parentDir( examplePath );
    std::string parentRoot = parentDir( root );
    if ( isRobustnessRoot( parentRoot ) )
        root = parentRoot;

    return root;
}

std::vector<std::pair<std::string, std::string>> collectExamples( std::string &root )
{
    std::vector<std::pair<std::string, std::string>> examples;
    bool isRobust = isRobustnessRoot( root );
    for ( auto &name : listDir( root ) )
    {
        std::string fullPath;
        if ( isRobust )
        {
            if ( name.find( "label" ) == std::string::npos )
                continue;
            fullPath = root + "/" + name + "/eps02.txt";
        }
        else if ( name.size() >= 4 && name.compare( name.size() - 4, 4, ".txt" ) == 0 )
            fullPath = root + "/" + name;

        else if ( name.size() >= 7 && name.compare( name.size() - 7, 7, ".vnnlib" ) == 0 )
            fullPath = root + "/" + name;
        else
            continue;
        std::string id;
        extractID( fullPath, id );
        examples.emplace_back( fullPath, id );
    }
    return examples;
}

std::vector<std::pair<std::string, std::string>>
randomSample( const std::vector<std::pair<std::string, std::string>> &all, int M )
{
    int N = (int)all.size();
    M = std::min( M, N );
    std::unordered_set<int> picks;
    while ( (int)picks.size() < M )
    {
        picks.insert( RandomGlobals::instance().randInt( 0, N - 1 ) );
    }
    std::vector<std::pair<std::string, std::string>> sampled;
    sampled.reserve( M );
    for ( int idx : picks )
        sampled.push_back( all[idx] );
    return sampled;
}


void trainAgentOnExamples( Options *options,
                           const std::vector<std::pair<std::string, std::string>> &examples,
                           std::unique_ptr<Agent> &agent )
{
    unsigned DQN_epochs = options->getInt( Options::DQN_EPOCHS );
    unsigned learnGuidedSteps = options->getInt( Options::DQN_GUIDED_STEPS );
    double epsilon = GlobalConfiguration::DQN_EPSILON_START;

    int N = options->getInt( Options::DQN_NUM_DEMO_EXAMPLES );
    N = std::min( N, (int)examples.size() );
    std::unordered_set<int> demoIndices;
    while ( demoIndices.size() < (size_t)N )
    {
        demoIndices.insert( RandomGlobals::instance().randInt( 0, examples.size() - 1 ) );
    }
    std::vector<std::pair<std::string, std::string>> demos;
    demos.reserve( N );
    for ( int idx : demoIndices )
        demos.push_back( examples[idx] );

    agent = nullptr;

    // 1) COLLECT DEMONSTRATION TRAJECTORIES
    DQN_LOG( "=== COLLECTING DEMOS ===\n" );
    GlobalConfiguration::DON_TRAINING_PHASE = 0;
    int numRepeats = 2;
    for ( auto &ex : demos )
    {
        for ( auto iter = 0; iter < numRepeats; iter++ )
        {
            options->setString( Options::PROPERTY_FILE_PATH, ex.first );
            // pseudo-impact
            GlobalConfiguration::DQN_FORCED_HEURISTIC =
                GlobalConfiguration::GuidedHeuristic::PSEUDO_IMPACT;
            agent = Marabou().trainDQNAgent( epsilon, ex.second, std::move( agent ) );
            // polarity
            GlobalConfiguration::DQN_FORCED_HEURISTIC =
                GlobalConfiguration::GuidedHeuristic::POLARITY;
            agent = Marabou().trainDQNAgent( epsilon, ex.second, std::move( agent ) );
            // BaBsr
            GlobalConfiguration::DQN_FORCED_HEURISTIC =
                GlobalConfiguration::GuidedHeuristic::BABS_R;
            agent = Marabou().trainDQNAgent( epsilon, ex.second, std::move( agent ) );
        }
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
        auto &ex = examples[epoch % examples.size()];
        options->setString( Options::PROPERTY_FILE_PATH, ex.first );
        agent = Marabou().trainDQNAgent( epsilon, ex.second, std::move( agent ) );
        epsilon = std::max( GlobalConfiguration::DQN_EPSILON_END,
                            epsilon * GlobalConfiguration::DQN_EPSILON_DECAY );
    }
    if ( agent != nullptr )
    {
        const auto path = options->getString( Options::DQN_AGENT_NETWORKS_PATH );
        const std::string filePath = std::string( path.ascii() ) + "/agent";
        agent->saveNetworks( filePath );
    }
}


void setRandomSeed()
{
    // unsigned seedVal = static_cast<unsigned>( std::time( nullptr ) );
    RandomGlobals::instance().seed( 1 );
}

int runRobustnessProperties( Options *options, const std::string &examplePath, bool agent )
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
        std::string currentExampleID;
        extractRobustnessExampleID( fullCurrentExamplePath, currentExampleID );
        size_t eps_pos = fullCurrentExamplePath.find( "eps" ) + 3;
        std::string eps_id = fullCurrentExamplePath.substr( eps_pos, 2 );
        options->setString( Options::PROPERTY_FILE_PATH, fullCurrentExamplePath );
        struct timespec startTime = TimeUtils::sampleMicro();
        if ( agent )
        {
            DQN_LOG( Stringf( "Start running trained agent with example: %s  ",
                              currentExampleID.c_str() )
                         .ascii() );
            Marabou().runTrainedAgent();
        }
        else
            Marabou().run();
        struct timespec endTime = TimeUtils::sampleMicro();
        unsigned long long totalTime = TimeUtils::timePassed( startTime, endTime );
        DQN_LOG( Stringf( "Done solving. Time : %llu milli. \n", totalTime / 1000 ).ascii() );
    }
    return 0;
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
            std::string exampleID;
            std::string examplePath =
                Options::get()->getString( Options::PROPERTY_FILE_PATH ).ascii();
            extractID( examplePath, exampleID );
            bool isRobust = isRobustnessRoot( parentDir( parentDir( examplePath ) ) );
            const std::string outDir = options->getString( Options::DQN_OUTPUT_FILE_PATH ).ascii();
            const std::string summaryFile = outDir + "/" + exampleID + ".txt";
            options->setString( Options::SUMMARY_FILE, summaryFile );
            if ( mode == 1 ) // TRAIN MODE
            {
                auto root = computeRootDir( examplePath );
                if ( root.empty() || !isDir( root ) )
                {
                    std::cerr << "Error: cannot determine root from '" << examplePath << "'\n";
                    return 1;
                }
                auto allExamples = collectExamples( root );
                auto sampled = randomSample(
                    allExamples, options->getInt( Options::DQN_NUM_TRAINING_EXAMPLES ) );

                options->setString( Options::SPLITTING_STRATEGY, "DQN-agent" );
                std::unique_ptr<Agent> agent;
                trainAgentOnExamples( options, sampled, agent );
            }
            else if ( mode == 2 ) // RUN MODE
            {
                options->setString( Options::SPLITTING_STRATEGY, "DQN-agent" );
                std::string agentPath =
                    options->getString( Options::DQN_AGENT_NETWORKS_PATH ).ascii();
                if ( !std::ifstream( agentPath + "_local.pth" ) ||
                     !std::ifstream( agentPath + "_target.pth" ) )
                {
                    std::cout << "trained agent network does not exist.\n";
                    return 0;
                }
                if ( isRobust )
                    return runRobustnessProperties( options, examplePath, true );
                Marabou().runTrainedAgent();
            }
            else // NO DQN MODE
            {
                if ( isRobust )
                    return runRobustnessProperties( options, examplePath, false );
                Marabou().run();
            }
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
