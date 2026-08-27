#include "functions.h"
#include <iostream>
#include <unistd.h>

using namespace std;

// checking if the command is one of the built-ins that we made ourselves
bool isBuiltin(const string &cmd)
{
    if (cmd == "cd" || cmd == "echo" || cmd == "pwd" || cmd == "ls" ||
        cmd == "pinfo" || cmd == "search" || cmd == "history" || cmd == "jobs")
        return true;

    return false;
}

// calling the right function for a built-in command
void runBuiltin(const vector<string> &tokens)
{
    string command = tokens[0];

    if (command == "cd")
        cdCommand(tokens);
    else if (command == "echo")
        echoCommand(tokens);
    else if (command == "pwd")
        pwdCommand();
    else if (command == "ls")
        lsCommand(tokens);
    else if (command == "pinfo")
        pinfoCommand(tokens);
    else if (command == "search")
        searchCommand(tokens);
    else if (command == "history")
        historyCommand(tokens);
    else if (command == "jobs")
        jobsCommand();
}

// built-ins run inside the shell itself, so to give them redirection we
// change stdin/stdout for a moment and then put them back afterwards
static void runBuiltinWithRedirection(vector<string> &cmdtokens, Redirection &redir)
{
    int savedIn = dup(STDIN_FILENO);
    int savedOut = dup(STDOUT_FILENO);
    bool ok = true;

    if (!redir.inputfile.empty())
    {
        int fd = open(redir.inputfile.c_str(), O_RDONLY);
        if (fd < 0)
        {
            perror(("input file " + redir.inputfile).c_str());
            ok = false;
        }
        else
        {
            dup2(fd, STDIN_FILENO);
            close(fd);
        }
    }

    if (ok && !redir.outputfile.empty())
    {
        int fd;
        if (redir.append)
            fd = open(redir.outputfile.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        else
            fd = open(redir.outputfile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);

        if (fd < 0)
        {
            perror(("output file " + redir.outputfile).c_str());
            ok = false;
        }
        else
        {
            dup2(fd, STDOUT_FILENO);
            close(fd);
        }
    }

    if (ok)
        runBuiltin(cmdtokens);

    cout.flush(); // making sure everything is written before we switch back

    dup2(savedIn, STDIN_FILENO);
    dup2(savedOut, STDOUT_FILENO);
    close(savedIn);
    close(savedOut);
}

// decides what to do with one command
void Runcmd(vector<string> &tokens)
{
    if (tokens.empty())
        return;

    bool background = false;
    if (tokens.back() == "&")
    {
        background = true;
        tokens.pop_back();
    }

    if (tokens.empty())
        return;

    // if there is a '|' in it then the pipeline code takes over
    if (PipelineHandle(tokens))
        return;

    if (isBuiltin(tokens[0]))
    {
        vector<string> cmdtokens = tokens;
        Redirection redir = redirectionParsing(cmdtokens);

        if (redir.error)
            return;

        if (cmdtokens.empty())
        {
            cerr << "There is no such command specified" << endl;
            return;
        }

        if (redir.inputfile.empty() && redir.outputfile.empty())
            runBuiltin(cmdtokens); // nothing special, just run it
        else
            runBuiltinWithRedirection(cmdtokens, redir);

        return;
    }

    systemCommand(tokens, background);
}
