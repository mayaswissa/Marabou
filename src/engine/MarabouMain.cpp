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

void extractExampleID( std::string &examplePath, std::string &exampleID )
{
    examplePath = Options::get()->getString( Options::PROPERTY_FILE_PATH ).ascii();
    size_t ex_pos = examplePath.find( "ex_" ) + 3;
    size_t label_pos = examplePath.find( "_label_" ) + 7;
    size_t eps_pos = examplePath.find( "eps" ) + 3;
    std::string ex_id = examplePath.substr( ex_pos, 4 );
    std::string label_id = examplePath.substr( label_pos, 1 );
    std::string eps_id = examplePath.substr( eps_pos, 2 );
    exampleID = ex_id + label_id + eps_id;
}
void generateRandomSeeds( int &numSeeds, Vector<int> &seeds )
{
    auto baseSeed = 0;
    numSeeds = 5;
    srand( baseSeed );

    for ( int i = 0; i < numSeeds; i++ )
    {
        int new_seed = ( rand() % 1000 ) + 1;
        seeds.append( new_seed );
    }
}
void trainAgentOnExample( Options *options,
                          const std::string &examplePath,
                          const std::string &exampleID,
                          const unsigned epochs,
                          std::unique_ptr<Agent> &agent,
                          int *numSplits,
                          std::ofstream &outputTxtFile )
{
    options->setString( Options::PROPERTY_FILE_PATH, examplePath );
    double epsilon = GlobalConfiguration::DQN_EPSILON_START;
    agent = nullptr;
    if ( outputTxtFile.is_open() )
    {
        outputTxtFile << "\n\t splits in each episode : \n\t\t";

        for ( unsigned int episode = 0; episode < epochs; ++episode )
        {
            int currentNumSplits = 0;
            agent = Marabou().runAgentTraining(
                epsilon, exampleID, true, std::move( agent ), &currentNumSplits );
            epsilon = std::max( GlobalConfiguration::DQN_EPSILON_END,
                                epsilon * GlobalConfiguration::DQN_EPSILON_DECAY );
            if ( outputTxtFile.is_open() )
                outputTxtFile << currentNumSplits << ", ";
            *numSplits += currentNumSplits;
            outputTxtFile.flush();
            agent->schedulersStep();
        }

        if ( agent != nullptr &&
             *numSplits < static_cast<int>( GlobalConfiguration::DQN_BATCH_SIZE ) )
        {
            const auto path = Options::get()->getString( Options::DQN_AGENT_NETWORKS_PATH );
            std::string filePath = std::string( path.ascii() ) + "/trainedAgent_" + exampleID;
            agent->saveNetworks( filePath );
        }

        outputTxtFile << "\n";
    }
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
                std::string examplePath;
                std::string trainedExampleID;
                extractExampleID( examplePath, trainedExampleID );

                unsigned epochs = 30;
                std::ostringstream currentRunFile;

                auto txtOutputFilePath = Options::get()->getString( Options::DQN_OUTPUT_FILE_PATH );
                currentRunFile << std::string( txtOutputFilePath.ascii() ) << "/"
                               << trainedExampleID << ".txt";
                std::ofstream outFile( currentRunFile.str(), std::ios::out | std::ios::app );
                if ( !outFile )
                {
                    std::cerr << "Failed to open " << currentRunFile.str() << "\n";
                    return 1;
                }

                struct timespec startTraining = TimeUtils::sampleMicro();
                int numSplits = 0;
                outFile << "Example : " << trainedExampleID << "\n";
                DQN_LOG( Stringf( "Start training agent on example: %s  ",
                                  trainedExampleID.c_str())
                             .ascii() );
                std::unique_ptr<Agent> agent;
                trainAgentOnExample( options,
                                     examplePath,
                                     trainedExampleID,
                                     epochs,
                                     agent,
                                     &numSplits,
                                     outFile );

                struct timespec endTraining = TimeUtils::sampleMicro();
                unsigned long long totalTraining =
                    TimeUtils::timePassed( startTraining, endTraining );
                outFile << "\t Done training"
                        << ". Time : " << totalTraining << " Splits : " << numSplits
                        << "\n";
                outFile << std::flush;
                outFile << "\n";

                outFile.close();
                DQN_LOG( Stringf( "Done training. Time : %llu milli. \n",
                                  totalTraining / 1000 )
                             .ascii() );
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
