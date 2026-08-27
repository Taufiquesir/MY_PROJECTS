#include "functions.h"
#include <iostream>

using namespace std;

// list of all the background and stopped processes
static vector<Job> JobList;

void add_job(pid_t pid, const string &name, bool stopped)
{
    // if the same pid is already there then we only update it
    for (size_t i = 0; i < JobList.size(); i++)
    {
        if (JobList[i].pid == pid)
        {
            JobList[i].stopped = stopped;
            return;
        }
    }

    Job j;
    j.pid = pid;
    j.name = name;
    j.stopped = stopped;
    JobList.push_back(j);
}

void remove_job(pid_t pid)
{
    for (size_t i = 0; i < JobList.size(); i++)
    {
        if (JobList[i].pid == pid)
        {
            JobList.erase(JobList.begin() + i);
            return;
        }
    }
}

// gives back the command name of a job, used when we print the messages
string job_name(pid_t pid)
{
    for (size_t i = 0; i < JobList.size(); i++)
    {
        if (JobList[i].pid == pid)
            return JobList[i].name;
    }
    return "";
}

void jobsCommand()
{
    for (size_t i = 0; i < JobList.size(); i++)
    {
        cout << "[" << (i + 1) << "] ";
        if (JobList[i].stopped)
            cout << "Stopped   ";
        else
            cout << "Running   ";
        cout << JobList[i].name << " [" << JobList[i].pid << "]" << endl;
    }
}
