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

void trainAgentOnExample( Options *options,
                          const std::string &examplePath,
                          const std::string &exampleID,
                          std::unique_ptr<Agent> &agent,
                          int *numSplits,
                          std::ofstream &outputTxtFile )
{
    unsigned epochs = options->getInt( Options::DQN_EPOCHS );
    options->setString( Options::PROPERTY_FILE_PATH, examplePath );
    double epsilon = GlobalConfiguration::DQN_EPSILON_START;
    agent = nullptr;
    if ( outputTxtFile.is_open() )
    {
        outputTxtFile << "\n\t results of each episode : \n";
        outputTxtFile << std::flush;
        for ( unsigned int episode = 0; episode < epochs; ++episode )
        {
            int currentNumSplits = 0;
            agent = Marabou().trainDQNAgent(
                epsilon, exampleID, std::move( agent ), &currentNumSplits );
            epsilon = std::max( GlobalConfiguration::DQN_EPSILON_END,
                                epsilon * GlobalConfiguration::DQN_EPSILON_DECAY );

            *numSplits += currentNumSplits;
            outputTxtFile.flush();
        }

        if ( agent != nullptr &&
             *numSplits >
                 static_cast<int>( Options::get()->getInt( Options::DQN_BATCH_SIZE ) * 20 ) )
        {
            const auto path = options->getString( Options::DQN_AGENT_NETWORKS_PATH );
            std::string filePath = std::string( path.ascii() ) + "/" + exampleID;
            agent->saveNetworks( filePath );
            outputTxtFile << "agent network has been saved. Path: " << filePath;
            outputTxtFile << std::flush;
        }

        outputTxtFile << "\n";
    }
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

            std::ostringstream currentRunFile;
            std::string examplePath;
            std::string exampleID;
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
            outFile << "Example : " << exampleID << "Network : " << network << ".\n";
            if ( mode == 1 )
            {
                // train
                struct timespec startTraining = TimeUtils::sampleMicro();
                int numSplits = 0;
                outFile << std::flush;
                DQN_LOG(
                    Stringf( "Start training agent on example: %s  ", exampleID.c_str() ).ascii() );
                std::unique_ptr<Agent> agent;
                trainAgentOnExample( options, examplePath, exampleID, agent, &numSplits, outFile );

                struct timespec endTraining = TimeUtils::sampleMicro();
                unsigned long long totalTraining =
                    TimeUtils::timePassed( startTraining, endTraining );
                outFile << "\t Done training"
                        << ". Time : " << totalTraining << " Splits : " << numSplits << "\n";
            }
            else if ( mode == 2 )
            {
                // run
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
                auto examples = listDir( root );
                for ( auto &currentExample : examples )
                {
                    if ( currentExample.size() < 4 ||
                         currentExample.substr( currentExample.size() - 4 ) != ".txt" )
                        continue;
                    std::string fullCurrentExamplePath = root + "/" + currentExample;
                    std::string currentExampleID;
                    extractExampleID( fullCurrentExamplePath, currentExampleID );
                    size_t eps_pos = fullCurrentExamplePath.find( "eps" ) + 3;
                    std::string eps_id = fullCurrentExamplePath.substr( eps_pos, 2 );
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
                    unsigned long long totalTraining = TimeUtils::timePassed(
                        startTime, endTime );

                    DQN_LOG( Stringf( "Done solving. Time : %llu milli. \n", totalTraining / 1000 )
                                 .ascii() );
                }
            }
            else
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
                    extractExampleID( fullCurrentExamplePath, currentExampleID );
                    size_t eps_pos = fullCurrentExamplePath.find( "eps" ) + 3;
                    std::string eps_id = fullCurrentExamplePath.substr( eps_pos, 2 );
                    options->setString( Options::PROPERTY_FILE_PATH, fullCurrentExamplePath );
                    struct timespec startTime = TimeUtils::sampleMicro();
                    outFile << "epsilon : " << eps_id << "\n";
                    outFile << std::flush;
                    Marabou().run();
                    struct timespec endTime = TimeUtils::sampleMicro();
                    unsigned long long totalTraining = TimeUtils::timePassed(
                        startTime, endTime );
                    DQN_LOG( Stringf( "Done solving. Time : %llu milli. \n", totalTraining / 1000 )
                                 .ascii() );

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
