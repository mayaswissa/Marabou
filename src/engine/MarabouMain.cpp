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
    size_t img_pos = examplePath.find("image");
    size_t tgt_pos = examplePath.find("_target");
    size_t eps_pos = examplePath.find("_epsilon");
    size_t txt_pos = examplePath.find(".txt");

    if (img_pos == std::string::npos || tgt_pos == std::string::npos ||
        eps_pos == std::string::npos )
    {
        exampleID = "";
        return;
    }

    size_t img_num_start = img_pos + 5; // after "image"
    size_t img_num_end = tgt_pos;
    std::string image = examplePath.substr(img_num_start, img_num_end - img_num_start);

    size_t tgt_num_start = tgt_pos + 7; // after "_target"
    size_t tgt_num_end = eps_pos;
    std::string target = examplePath.substr(tgt_num_start, tgt_num_end - tgt_num_start);

    size_t eps_num_start = eps_pos + 10; // after "_epsilon"
    size_t eps_num_end = txt_pos;
    std::string epsilon = examplePath.substr(eps_num_start, eps_num_end - eps_num_start);
    exampleID = image + target + epsilon;
}
void extractNetworkName(std::string & network )
{
    String networkFilePath = Options::get()->getString( Options::INPUT_FILE_PATH );
    std::string networkPath = static_cast<std::string>(networkFilePath.ascii());
    size_t start = networkPath.find_last_of( '/' );
    size_t end = networkFilePath.find( ".nnet" );
    network = networkPath.substr( start + 1, end - start - 1 );
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
            std::ostringstream currentRunFile;
            std::string examplePath;
            std::string exampleID;
            extractExampleID( examplePath, exampleID );
            auto txtOutputFilePath = Options::get()->getString( Options::DQN_OUTPUT_FILE_PATH );
            std::string network;
            extractNetworkName(network);
            currentRunFile << std::string( txtOutputFilePath.ascii() ) << network << "_" << exampleID << ".txt";
            std::ofstream outFile( currentRunFile.str(), std::ios::out | std::ios::app );

            if ( outFile.is_open() )
            {
                int numSplits = 0;
                String runResult;
                struct timespec startCurrExample = TimeUtils::sampleMicro();
                String exitCode;
                Marabou().run( &numSplits, exitCode );
                struct timespec endCurrExample = TimeUtils::sampleMicro();
                unsigned long long totalRunCurrExample =
                    TimeUtils::timePassed( startCurrExample, endCurrExample );

                auto totalMilli = std::to_string( totalRunCurrExample / 1000 ).c_str();
                outFile << "\n";
                outFile << ", Example ID: " << exampleID << ", numSplits:" << numSplits
                        << ", Time : " << totalMilli << " milli"
                        << ", Exit code: " << exitCode.ascii() << "\n";
                outFile << std::flush;
                outFile << "\n";
                outFile.close();
            }
            else
            {
                std::cerr << "Failed to open file: " << currentRunFile.str() << std::endl;
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
