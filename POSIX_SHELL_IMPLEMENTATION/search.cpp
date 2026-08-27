#include "functions.h"
#include <iostream>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <cstring>

using namespace std;

static bool RecurSearch(const string &basePath, const string &target) // It will search for file and directory
{
    DIR *dir = opendir(basePath.c_str());
    if (!dir)
    {
        return false;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        string name = entry->d_name;
        if (name == "." || name == "..") // It will Skip . and ..
            continue;

        string fullPath = basePath + "/" + name;

        if (name == target)
        {
            closedir(dir);
            return true;
        }

        // lstat is used instead of stat so that symbolic links are not
        // followed, otherwise a link loop can make this go on forever
        struct stat statbuff;
        if (lstat(fullPath.c_str(), &statbuff) == 0)
        {
            if (S_ISDIR(statbuff.st_mode))
            {
                if (RecurSearch(fullPath, target))
                {
                    closedir(dir);
                    return true;
                }
            }
        }
    }

    closedir(dir);
    return false;
}

void searchCommand(const vector<string> &tokens)
{
    if (tokens.size() != 2)
    {
        cout << "Usage: search <filename>" << endl;
        return;
    }

    string target = tokens[1];

    char cwd[1024];
    if (!getcwd(cwd, sizeof(cwd)))
    {
        perror("search: getcwd is not working");
        return;
    }

    string currentDir = cwd;
    if (RecurSearch(currentDir, target))
        cout << "True" << endl;
    else
        cout << "False" << endl;
}
