#include "functions.h"
#include <iostream>
#include <unistd.h>

using namespace std;

void pwdCommand()
{
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL)
    {
        cout << cwd << endl;
    }
    else
    {
        perror("pwd is not working");
    }
}
