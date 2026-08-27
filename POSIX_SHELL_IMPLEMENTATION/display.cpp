#include "functions.h"
#include <iostream>
#include <unistd.h>
#include <cstring>
#include <pwd.h>

using namespace std;

string displayPrompt()
{
    struct passwd *pw = getpwuid(getuid()); // fetching and storing username
    string username;
    if (pw != NULL)
        username = pw->pw_name;
    else
        username = "unknown";

    char hostname[1024];
    string sysname;
    if (gethostname(hostname, sizeof(hostname)) == 0) // fetching and storing hostname
    {
        hostname[sizeof(hostname) - 1] = '\0'; // making sure it ends properly
        sysname = hostname;
    }
    else
    {
        sysname = "unknown";
    }

    char cwd[1024];
    string currDir;
    if (getcwd(cwd, sizeof(cwd)) != NULL) // fetching the current directory
        currDir = cwd;
    else
        currDir = "?";

    // the directory in which the shell was started is shown as ~
    if (!MyHomeDir.empty() && currDir.compare(0, MyHomeDir.length(), MyHomeDir) == 0)
    {
        // it must be the home itself or something inside it, not just a name
        // which starts with the same letters
        if (currDir.length() == MyHomeDir.length() || currDir[MyHomeDir.length()] == '/')
            currDir.replace(0, MyHomeDir.length(), "~");
    }

    string prompt = "<" + username + "@" + sysname + ":" + currDir + "> ";
    return prompt;
}
