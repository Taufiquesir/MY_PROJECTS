#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <vector>
#include <string>
#include <sys/types.h> // importing pid_t
#include <fcntl.h>

extern std::string MyHomeDir;
extern std::string PrevDir;

// holds the file names found after < , > and >>
struct Redirection
{
    std::string inputfile;
    std::string outputfile;
    bool append = false;
    bool error = false;   // true if the syntax was wrong
};

// one background or stopped process
struct Job
{
    pid_t pid;
    std::string name;
    bool stopped;
};

// parser.cpp
std::vector<std::string> SplitBySemicolon(const std::string &input);
std::vector<std::string> InputParsing(const std::string &input);

// executor.cpp
void Runcmd(std::vector<std::string> &tokens);
bool isBuiltin(const std::string &cmd);
void runBuiltin(const std::vector<std::string> &tokens);

// each command has got its own file
void cdCommand(const std::vector<std::string> &tokens);      // cd.cpp
void echoCommand(const std::vector<std::string> &tokens);    // echo.cpp
void pwdCommand();                                           // pwd.cpp
void lsCommand(const std::vector<std::string> &tokens);      // ls.cpp
void pinfoCommand(const std::vector<std::string> &tokens);   // pinfo.cpp
void searchCommand(const std::vector<std::string> &tokens);  // search.cpp
void historyCommand(const std::vector<std::string> &tokens); // history.cpp
void jobsCommand();                                          // jobs.cpp

// system.cpp
void systemCommand(const std::vector<std::string> &tokens, bool background = false);

// redirection.cpp
Redirection redirectionParsing(std::vector<std::string> &cmdtokens);

// pipe.cpp
bool PipelineHandle(std::vector<std::string> &tokens);

// signal.cpp
void init_signals();
void set_foreground_pid(pid_t pid);
pid_t get_foreground_pid();
void set_foreground_name(const std::string &name);
std::string get_foreground_name();

// jobs.cpp
void add_job(pid_t pid, const std::string &name, bool stopped);
void remove_job(pid_t pid);
std::string job_name(pid_t pid);

// autocomplete.cpp
void setup_autocomplete();

// history.cpp
void LoadHistory();
void add_to_history(const std::string &command);

// display.cpp
std::string displayPrompt();

#endif
