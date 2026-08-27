#include "functions.h"
#include <iostream>
#include <unistd.h>
#include <sys/wait.h>
#include <cstring>
#include <sys/stat.h>
#include <csignal>

using namespace std;

// runs a pipeline, it also takes care of the redirection inside it.
// returns false when the command has no '|' in it, then the normal code runs.
bool PipelineHandle(vector<string> &tokens)
{
    // first checking if there is a pipe at all
    bool hasPipe = false;
    for (size_t i = 0; i < tokens.size(); i++)
    {
        if (tokens[i] == "|")
            hasPipe = true;
    }
    if (!hasPipe)
        return false;

    // splitting the tokens into the separate commands on '|'
    vector<vector<string> > commands;
    vector<string> current;

    for (size_t i = 0; i < tokens.size(); i++)
    {
        if (tokens[i] == "|")
        {
            if (current.empty())
            {
                cerr << "Syntax error: command should not be empty before/after '|'" << endl;
                return true;
            }
            commands.push_back(current);
            current.clear();
        }
        else
        {
            current.push_back(tokens[i]);
        }
    }

    if (current.empty())
    {
        cerr << "Syntax error: command should not be empty before/after '|'" << endl;
        return true;
    }
    commands.push_back(current);

    int n = commands.size();

    // taking out the redirection of every command before we start forking,
    // so that a syntax mistake is caught early
    vector<Redirection> redirs;
    for (int i = 0; i < n; i++)
    {
        Redirection r = redirectionParsing(commands[i]);
        if (r.error)
            return true;
        if (commands[i].empty())
        {
            cerr << "There is no such command specified" << endl;
            return true;
        }
        redirs.push_back(r);
    }

    // making one pipe between every pair of commands
    int pipecount = n - 1;
    vector<int> pipefds(2 * pipecount);

    for (int i = 0; i < pipecount; i++)
    {
        if (pipe(&pipefds[i * 2]) < 0)
        {
            perror("pipe");
            return true;
        }
    }

    vector<pid_t> pids;

    for (int i = 0; i < n; i++)
    {
        pid_t pid = fork();

        if (pid == 0) // child
        {
            setpgid(0, 0);
            signal(SIGINT, SIG_DFL);
            signal(SIGTSTP, SIG_DFL);

            // Input side, a file given with '<' is more important than the pipe
            if (!redirs[i].inputfile.empty())
            {
                int fd = open(redirs[i].inputfile.c_str(), O_RDONLY);
                if (fd < 0)
                {
                    perror(("input file " + redirs[i].inputfile).c_str());
                    _exit(1);
                }
                dup2(fd, STDIN_FILENO);
                close(fd);
            }
            else if (i > 0)
            {
                dup2(pipefds[(i - 1) * 2], STDIN_FILENO); // read end of previous pipe
            }

            // Output side, a file given with '>' is more important than the pipe
            if (!redirs[i].outputfile.empty())
            {
                int fd;
                if (redirs[i].append)
                    fd = open(redirs[i].outputfile.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
                else
                    fd = open(redirs[i].outputfile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);

                if (fd < 0)
                {
                    perror(("output file " + redirs[i].outputfile).c_str());
                    _exit(1);
                }
                dup2(fd, STDOUT_FILENO);
                close(fd);
            }
            else if (i < n - 1)
            {
                dup2(pipefds[i * 2 + 1], STDOUT_FILENO); // write end of our pipe
            }

            // the child does not need any of the pipe fds anymore
            for (int j = 0; j < 2 * pipecount; j++)
                close(pipefds[j]);

            // built-ins are also allowed inside a pipeline
            if (isBuiltin(commands[i][0]))
            {
                runBuiltin(commands[i]);
                cout.flush();
                _exit(0);
            }

            vector<char *> exec_args;
            for (size_t k = 0; k < commands[i].size(); k++)
                exec_args.push_back(const_cast<char *>(commands[i][k].c_str()));
            exec_args.push_back(NULL);

            execvp(exec_args[0], exec_args.data());

            perror(commands[i][0].c_str());
            _exit(1);
        }
        else if (pid < 0)
        {
            perror("fork");
            for (int j = 0; j < 2 * pipecount; j++)
                close(pipefds[j]);
            return true;
        }
        else
        {
            pids.push_back(pid);
        }
    }

    // the parent must close every pipe end, otherwise the readers will
    // never see the end of the input and will keep waiting
    for (int j = 0; j < 2 * pipecount; j++)
        close(pipefds[j]);

    // waiting for all the commands of the pipeline to finish.
    // the last command is kept as the foreground one, so that Ctrl+C works
    // and the SIGCHLD handler does not reap these children by mistake
    set_foreground_pid(pids[pids.size() - 1]);
    set_foreground_name(commands[n - 1][0]);

    for (size_t i = 0; i < pids.size(); i++)
    {
        int status;
        waitpid(pids[i], &status, 0);
    }

    set_foreground_pid(-1);
    set_foreground_name("");

    return true;
}
