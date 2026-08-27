#include "functions.h"
#include <iostream>
#include <unistd.h>
#include <cstring>

using namespace std;

void cdCommand(const vector<string> &tokens)
{
    if (tokens.size() > 2)
    {
        cout << "Invalid arguments" << endl;
        return;
    }

    char oldCwd[1024];
    if (getcwd(oldCwd, sizeof(oldCwd)) == NULL)
    {
        perror("cd");
        return;
    }

    string targetDir;
    if (tokens.size() == 1)
    {
        targetDir = MyHomeDir; // no argument means go into home
    }
    else
    {
        targetDir = tokens[1];

        if (targetDir == "~") // goto Home directory
        {
            targetDir = MyHomeDir;
        }
        else if (targetDir.compare(0, 2, "~/") == 0) // paths like ~/test
        {
            targetDir = MyHomeDir + targetDir.substr(1);
        }
        else if (targetDir == "-") // goto Previous directory
        {
            if (PrevDir.empty())
            {
                cout << "cd: previous directory is not set" << endl;
                return;
            }
            targetDir = PrevDir;
            cout << targetDir << endl;
        }
        // "." and ".." are normal paths so chdir can handle them itself
    }

    if (chdir(targetDir.c_str()) != 0)
    {
        perror("cd failed");
    }
    else
    {
        PrevDir = oldCwd;
    }
}
