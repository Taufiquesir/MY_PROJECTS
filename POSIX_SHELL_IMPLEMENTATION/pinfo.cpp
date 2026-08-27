#include "functions.h"
#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>
#include <sys/types.h>
#include <sstream>
#include <cctype>

using namespace std;

// replaces the home directory part of a path with ~
static string tilde_path(const string &p)
{
    if (!MyHomeDir.empty() && p.compare(0, MyHomeDir.size(), MyHomeDir) == 0)
        return string("~") + p.substr(MyHomeDir.size());
    return p;
}

void pinfoCommand(const vector<string> &tokens)
{
    pid_t pid;

    if (tokens.size() == 1)
    {
        pid = getpid();
    }
    else if (tokens.size() == 2)
    {
        // checking that the argument is actually a number
        bool isnum = !tokens[1].empty();
        for (size_t i = 0; i < tokens[1].length(); i++)
        {
            if (!isdigit((unsigned char)tokens[1][i]))
                isnum = false;
        }
        if (!isnum)
        {
            cerr << "pinfo: PID is invalid" << endl;
            return;
        }

        try
        {
            pid = stoi(tokens[1]);
        }
        catch (...)
        {
            cerr << "pinfo: PID is invalid" << endl;
            return;
        }
    }
    else
    {
        cerr << "Usage: pinfo [PID]" << endl;
        return;
    }

    string stat_path = "/proc/" + to_string(pid) + "/stat";
    ifstream stat_file(stat_path);
    if (!stat_file)
    {
        cerr << "pinfo: process with pid " << pid << " not found" << endl;
        return;
    }

    string line;
    getline(stat_file, line);
    stat_file.close();

    // the process name is inside brackets and it can have spaces in it,
    // so we start reading the fields only after the last ')'
    size_t bracket = line.rfind(')');
    if (bracket == string::npos)
    {
        cerr << "pinfo: invalid stat format" << endl;
        return;
    }

    istringstream iss(line.substr(bracket + 1));
    vector<string> fields;
    string token;
    while (iss >> token)
        fields.push_back(token);

    // fields[0] is now the state, which is field 3 in the original file.
    // so field number N of the file is fields[N - 3] here.
    if (fields.size() < 21)
    {
        cerr << "pinfo: invalid stat format" << endl;
        return;
    }

    string state = fields[0];        // field 3
    long pgrp = stol(fields[2]);     // field 5
    long tpgid = stol(fields[4]);    // field 7
    unsigned long vsize = stoul(fields[20]); // field 23

    // trying to read the real executable path
    string exe_path = "/proc/" + to_string(pid) + "/exe";
    char exe_buf[4096];
    ssize_t len = readlink(exe_path.c_str(), exe_buf, sizeof(exe_buf) - 1);

    string execPath;
    if (len != -1)
    {
        execPath = string(exe_buf, len);
    }
    else
    {
        // readlink can fail if we do not have permission, so we fall back
        // to reading the command line from /proc/<pid>/cmdline
        string cmd_path = "/proc/" + to_string(pid) + "/cmdline";
        ifstream cmd_file(cmd_path);
        if (cmd_file)
        {
            string cmdline;
            getline(cmd_file, cmdline, '\0'); // arguments are separated by null
            cmd_file.close();
            if (!cmdline.empty())
                execPath = cmdline;
        }
        if (execPath.empty())
            execPath = "N/A";
    }

    string formatted_state = state;

    // "+" is added when the process is in the foreground, that means its
    // process group is the same as the terminal's process group
    if (tpgid != -1 && tpgid == pgrp)
    {
        formatted_state += "+";
    }

    cout << "Process Status -- " << formatted_state << endl;
    cout << "memory -- " << vsize << " {Virtual Memory}" << endl;
    cout << "Executable Path -- " << tilde_path(execPath) << endl;
}
