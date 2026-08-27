#include "functions.h"
#include <iostream>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <cstring>
#include <csignal>

using namespace std;

void systemCommand(const vector<string> &tokens, bool background)
{
    if (tokens.empty())
        return;

    vector<string> cmdtokens = tokens;

    Redirection redir = redirectionParsing(cmdtokens);
    if (redir.error)
        return;

    if (cmdtokens.empty())
    {
        cerr << "There is no such command specified" << endl;
        return;
    }

    // the input file is checked here in the parent itself, so that the error
    // is shown by the shell before a new process is even created
    if (!redir.inputfile.empty())
    {
        int testfd = open(redir.inputfile.c_str(), O_RDONLY);
        if (testfd < 0)
        {
            perror(("input file " + redir.inputfile).c_str());
            return;
        }
        close(testfd);
    }

    vector<char *> cargs; // It will convert args to execvp format
    for (size_t i = 0; i < cmdtokens.size(); i++)
        cargs.push_back(const_cast<char *>(cmdtokens[i].c_str()));
    cargs.push_back(NULL);

    pid_t pid = fork();

    if (pid == 0) // It is child process
    {
        // the child is put in its own process group, this way Ctrl+C and
        // Ctrl+Z of the terminal do not hit the shell itself
        setpgid(0, 0);

        // the child should react to these signals in the normal way
        signal(SIGINT, SIG_DFL);
        signal(SIGTSTP, SIG_DFL);

        if (!redir.inputfile.empty()) // It will redirect Input
        {
            int fd = open(redir.inputfile.c_str(), O_RDONLY);
            if (fd < 0)
            {
                perror(("input file " + redir.inputfile).c_str());
                _exit(1);
            }
            dup2(fd, STDIN_FILENO);
            close(fd);
        }

        if (!redir.outputfile.empty()) // It will redirect Output
        {
            int fd;
            if (redir.append)
                fd = open(redir.outputfile.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
            else
                fd = open(redir.outputfile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);

            if (fd < 0)
            {
                perror(("output file " + redir.outputfile).c_str());
                _exit(1);
            }
            dup2(fd, STDOUT_FILENO);
            close(fd);
        }

        execvp(cargs[0], cargs.data());

        // we only reach here if execvp could not run the command
        perror(cmdtokens[0].c_str());
        _exit(1);
    }
    else if (pid > 0) // this is parent process
    {
        setpgid(pid, pid); // doing it here also, so that there is no race

        if (!background)
        {
            set_foreground_pid(pid);
            set_foreground_name(cmdtokens[0]);

            int status;
            // WUNTRACED so that we come back when the child is stopped by Ctrl+Z
            waitpid(pid, &status, WUNTRACED);

            set_foreground_pid(-1);
            set_foreground_name("");

            if (WIFSTOPPED(status))
            {
                // the job was stopped by Ctrl+Z, now it stays in the background
                add_job(pid, cmdtokens[0], true);
                cout << endl << cmdtokens[0] << " with pid " << pid
                     << " is stopped and sent to background" << endl;
            }
        }
        else
        {
            // printing the pid of the new background process
            add_job(pid, cmdtokens[0], false);
            cout << pid << endl;
        }
    }
    else
    {
        perror("fork is failed");
    }
}
