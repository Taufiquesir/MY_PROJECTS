#include "functions.h"
#include <readline/readline.h>
#include <readline/history.h>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <string>
#include <vector>

using namespace std;

// our own commands
static const char *builtinNames[] = {"cd", "echo", "pwd", "ls", "pinfo",
                                     "search", "history", "jobs", "exit"};

static vector<string> allCommands; // built-ins plus everything from PATH
static bool commandsLoaded = false;

// reads all the program names from the directories present in PATH,
// this is needed so that things like "ca<TAB>" can show cat, capsh etc.
static void loadCommands()
{
    if (commandsLoaded)
        return;

    for (size_t i = 0; i < sizeof(builtinNames) / sizeof(builtinNames[0]); i++)
        allCommands.push_back(builtinNames[i]);

    char *pathenv = getenv("PATH");
    if (pathenv != NULL)
    {
        string path = pathenv;
        size_t start = 0;

        // PATH has got the directories separated by ':'
        while (start <= path.length())
        {
            size_t pos = path.find(':', start);
            if (pos == string::npos)
                pos = path.length();

            string dirname = path.substr(start, pos - start);
            if (!dirname.empty())
            {
                DIR *dir = opendir(dirname.c_str());
                if (dir)
                {
                    struct dirent *entry;
                    while ((entry = readdir(dir)) != NULL)
                    {
                        string name = entry->d_name;
                        if (name == "." || name == "..")
                            continue;

                        // keeping only the files which we can actually run
                        string full = dirname + "/" + name;
                        if (access(full.c_str(), X_OK) == 0)
                            allCommands.push_back(name);
                    }
                    closedir(dir);
                }
            }

            start = pos + 1;
        }
    }

    // sorting and removing the duplicates, one program can be in two places
    sort(allCommands.begin(), allCommands.end());
    allCommands.erase(unique(allCommands.begin(), allCommands.end()), allCommands.end());

    commandsLoaded = true;
}

// readline calls this again and again until it returns NULL,
// every call has to give back one more matching command
static char *cmdGenerator(const char *text, int state)
{
    static size_t list_index;
    static size_t len;

    if (state == 0) // state 0 means this is a fresh search
    {
        loadCommands();
        list_index = 0;
        len = strlen(text);
    }

    while (list_index < allCommands.size())
    {
        const string &cmd = allCommands[list_index++];
        if (cmd.length() >= len && cmd.compare(0, len, text) == 0)
            return strdup(cmd.c_str());
    }

    return NULL;
}

// Generator for the file and directory names of the current directory
static char *filenameGntr(const char *text, int state)
{
    static vector<string> matches;
    static size_t index;

    if (state == 0)
    {
        matches.clear();
        index = 0;

        size_t len = strlen(text);

        DIR *dir = opendir(".");
        if (dir)
        {
            struct dirent *entry;
            while ((entry = readdir(dir)) != NULL)
            {
                string name = entry->d_name;
                if (name == "." || name == "..")
                    continue;

                if (name.length() >= len && name.compare(0, len, text) == 0)
                {
                    // putting a '/' after the directories so it is easy to see
                    struct stat st;
                    if (stat(name.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
                        matches.push_back(name + "/");
                    else
                        matches.push_back(name);
                }
            }
            closedir(dir);
        }

        sort(matches.begin(), matches.end());
    }

    if (index < matches.size())
        return strdup(matches[index++].c_str());

    return NULL;
}

// the first word of the line is a command, everything after it is a file name
static char **customCompletion(const char *text, int start, int)
{
    // this stops readline from doing its own file completion after ours
    rl_attempted_completion_over = 1;

    // checking if only spaces are there before this word, then it is the command
    bool isFirstWord = true;
    for (int i = 0; i < start; i++)
    {
        if (rl_line_buffer[i] != ' ' && rl_line_buffer[i] != '\t')
            isFirstWord = false;
    }

    if (isFirstWord)
        return rl_completion_matches(text, cmdGenerator);
    else
        return rl_completion_matches(text, filenameGntr);
}

void setup_autocomplete()
{
    rl_attempted_completion_function = customCompletion;
    rl_readline_name = (char *)"myshell";
    rl_bind_key('\t', rl_complete);
}
