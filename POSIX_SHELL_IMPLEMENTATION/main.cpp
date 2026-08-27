#include "functions.h"
#include <iostream>
#include <unistd.h>
#include <cctype>
#include <cstdlib>
#include <readline/readline.h>
#include <readline/history.h>

using namespace std;

std::string MyHomeDir;
std::string PrevDir;

// checks if the line has got only spaces and tabs in it
static bool isOnlyWhitespace(const string &input)
{
    for (size_t i = 0; i < input.length(); i++)
    {
        if (!isspace((unsigned char)input[i]))
            return false;
    }
    return true;
}

int main()
{
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL)
        MyHomeDir = cwd;

    PrevDir = MyHomeDir; // so that cd - works even on a fresh shell

    init_signals(); // Initializing and handling bg/fg signals
    setup_autocomplete();
    LoadHistory();

    while (true)
    {
        string prompt = displayPrompt(); // build prompt

        char *tmpInput = readline(prompt.c_str()); // readline is checking backspace, arrows

        if (!tmpInput)
        {
            cout << endl; // handling Ctrl-D here
            break;
        }

        string input(tmpInput);
        free(tmpInput);

        if (isOnlyWhitespace(input)) // it will ignore empty or whitespace
            continue;

        add_to_history(input); // adding history

        // splitting the whole line on ';' to get the separate commands
        vector<string> cmdlist = SplitBySemicolon(input);

        bool wantExit = false;
        for (size_t i = 0; i < cmdlist.size(); i++)
        {
            vector<string> tokens = InputParsing(cmdlist[i]);
            if (tokens.empty())
                continue;

            if (tokens[0] == "exit" || tokens[0] == "quit")
            {
                wantExit = true;
                break;
            }

            Runcmd(tokens);
        }

        if (wantExit)
            break;
    }

    return 0;
}
