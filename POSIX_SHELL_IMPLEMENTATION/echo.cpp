#include "functions.h"
#include <iostream>

using namespace std;

void echoCommand(const vector<string> &tokens)
{
    // just printing all the arguments with one space between them
    for (size_t i = 1; i < tokens.size(); i++)
    {
        if (i > 1)
            cout << " ";
        cout << tokens[i];
    }
    cout << endl;
}
