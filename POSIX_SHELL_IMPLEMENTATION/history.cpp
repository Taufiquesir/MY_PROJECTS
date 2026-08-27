#include "functions.h"
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <pwd.h>
#include <algorithm>
#include <readline/readline.h>
#include <readline/history.h>

using namespace std;

#define MAX_HISTORY 20
#define MAX_SHOW 10
#define HISTORY_FILE ".myshell_history"

vector<string> CmdHistory;

// the history file is kept in the user's real home so that it works
// across all the sessions
static string historyPath()
{
    struct passwd *pw = getpwuid(getuid());
    string home_dir = pw ? pw->pw_dir : ".";
    return home_dir + "/" + HISTORY_FILE;
}

// It will Load history from file
void LoadHistory()
{
    CmdHistory.clear();

    ifstream history_file(historyPath());
    if (history_file.is_open())
    {
        string line;
        while (getline(history_file, line))
        {
            if (!line.empty())
            {
                CmdHistory.push_back(line);
                if ((int)CmdHistory.size() > MAX_HISTORY)
                    CmdHistory.erase(CmdHistory.begin()); // oldest one goes out
            }
        }
        history_file.close();
    }

    // load into readline so that up/down arrow keys work
    clear_history();
    for (size_t i = 0; i < CmdHistory.size(); i++)
        add_history(CmdHistory[i].c_str());
}

// It is for Saving history
static void save_history()
{
    ofstream history_file(historyPath());
    if (history_file.is_open())
    {
        for (size_t i = 0; i < CmdHistory.size(); i++)
            history_file << CmdHistory[i] << endl;
        history_file.close();
    }
}

//  Add command to the history
void add_to_history(const string &command)
{
    // It will delete leading/trailing whitespace
    string trimmedCmd = command;
    size_t start = trimmedCmd.find_first_not_of(" \t");
    size_t end = trimmedCmd.find_last_not_of(" \t");

    if (start != string::npos && end != string::npos)
        trimmedCmd = trimmedCmd.substr(start, end - start + 1);
    else
        trimmedCmd = "";

    if (trimmedCmd.empty())
        return;

    CmdHistory.push_back(trimmedCmd);

    // if we cross the limit then the oldest command is overwritten
    while ((int)CmdHistory.size() > MAX_HISTORY)
        CmdHistory.erase(CmdHistory.begin());

    add_history(trimmedCmd.c_str());
    save_history();
}

void historyCommand(const vector<string> &tokens)
{
    int count = MAX_SHOW; // by default it shows the last 10 commands

    if (tokens.size() > 2)
    {
        cout << "Usage: history [number]" << endl;
        return;
    }

    if (tokens.size() == 2)
    {
        try
        {
            size_t pos;
            count = stoi(tokens[1], &pos);
            if (pos != tokens[1].length()) // something like "12abc"
            {
                cout << "history: invalid argument" << endl;
                return;
            }
            if (count <= 0)
            {
                cout << "history: count must be positive" << endl;
                return;
            }
            if (count > MAX_HISTORY)
                count = MAX_HISTORY;
        }
        catch (...)
        {
            cout << "history: invalid argument" << endl;
            return;
        }
    }

    int total = (int)CmdHistory.size();
    int start = max(0, total - count);
    for (int i = start; i < total; i++)
        cout << CmdHistory[i] << endl;
}
