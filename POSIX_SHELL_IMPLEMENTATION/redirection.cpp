#include "functions.h"
#include <iostream>

using namespace std;

Redirection redirectionParsing(vector<string> &cmdtokens)
{
    Redirection r;
    vector<string> args;

    for (size_t i = 0; i < cmdtokens.size(); i++)
    {
        if (cmdtokens[i] == "<")
        {
            if (i + 1 < cmdtokens.size())
            {
                r.inputfile = cmdtokens[++i];
            }
            else
            {
                cerr << "Syntax error: file name is missing after '<'" << endl;
                r.error = true;
                return r;
            }
        }
        else if (cmdtokens[i] == ">")
        {
            if (i + 1 < cmdtokens.size())
            {
                r.outputfile = cmdtokens[++i];
                r.append = false;
            }
            else
            {
                cerr << "Syntax error: file name is missing after '>'" << endl;
                r.error = true;
                return r;
            }
        }
        else if (cmdtokens[i] == ">>")
        {
            if (i + 1 < cmdtokens.size())
            {
                r.outputfile = cmdtokens[++i];
                r.append = true;
            }
            else
            {
                cerr << "Syntax error: file name is missing after '>>'" << endl;
                r.error = true;
                return r;
            }
        }
        else
        {
            args.push_back(cmdtokens[i]);
        }
    }

    cmdtokens = args;
    return r;
}
