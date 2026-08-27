#include "functions.h"
#include <iostream>
#include <csignal>
#include <unistd.h>
#include <sys/wait.h>
#include <cstring>

using namespace std;

static pid_t PIDfg = -1;       // storing PID of foreground for tracking
static string NameFg = "";     // name of the foreground command
static volatile sig_atomic_t waitingFg = 0; // 1 while the shell waits for a foreground job

// Getting and Setting Foreground PID.
void set_foreground_pid(pid_t pid)
{
    PIDfg = pid;
    // when a foreground job is running we tell the SIGCHLD handler not to
    // touch anything, otherwise it can reap that child before waitpid does
    waitingFg = (pid > 0) ? 1 : 0;
}

pid_t get_foreground_pid()
{
    return PIDfg;
}

void set_foreground_name(const string &name)
{
    NameFg = name;
}

string get_foreground_name()
{
    return NameFg;
}

// SIGCHLD Handler, it reaps the background children when they finish
static void handle_sigchld(int)
{
    int status;
    pid_t pid;

    // while a foreground job is running, the waitpid in system.cpp/pipe.cpp
    // is the one which should reap the children, so we stay away
    if (waitingFg)
        return;

    // WNOHANG is used so that this does not block the shell
    while ((pid = waitpid(-1, &status, WNOHANG)) > 0)
    {
        string name = job_name(pid);
        if (name.empty())
            name = "process";

        // write() is used here because printing with cout inside a signal
        // handler is not safe
        string msg = "\n" + name + " with pid " + to_string(pid) + " exited\n";
        ssize_t r = write(STDOUT_FILENO, msg.c_str(), msg.length());
        (void)r;

        remove_job(pid);
    }
}

// SIGINT Handler is used to handle Ctrl+C
static void handle_sigint(int)
{
    if (PIDfg > 0)
    {
        kill(PIDfg, SIGINT); // It will send SIGINT to foreground process
    }
    // if there is no foreground process then nothing happens to the shell
}

// SIGTSTP Handler (Ctrl+Z), it pushes the foreground job to background
static void handle_sigtstp(int)
{
    if (PIDfg > 0)
    {
        kill(PIDfg, SIGTSTP); // stopping the foreground process

        // now the job is kept in the list as a stopped one
        add_job(PIDfg, NameFg, true);
    }
    // if there is no foreground process then nothing happens to the shell
}

void init_signals()
{
    struct sigaction sa;

    // SIGCHLD is used for the purpose of background cleaning
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_sigchld;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART | SA_NOCLDSTOP; // NOCLDSTOP so stopped ones are not reported here
    if (sigaction(SIGCHLD, &sa, NULL) == -1)
    {
        perror("sigaction SIGCHLD");
    }

    // Foreground handling is being done here
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    if (sigaction(SIGINT, &sa, NULL) == -1) // Ctrl+C
    {
        perror("sigaction SIGINT");
    }

    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_sigtstp;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    if (sigaction(SIGTSTP, &sa, NULL) == -1) // Ctrl+Z
    {
        perror("sigaction SIGTSTP");
    }
}
