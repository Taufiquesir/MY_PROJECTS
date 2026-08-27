#include "functions.h"
#include <iostream>
#include <cstring>
#include <cctype>

using namespace std;

// splits the input line into separate commands on ';' using strtok
vector<string> SplitBySemicolon(const string &input)
{
    vector<string> parts;

    // strtok changes the buffer that is given to it, so we make a copy first
    char *buffer = new char[input.length() + 1];
    strcpy(buffer, input.c_str());

    char *piece = strtok(buffer, ";");
    while (piece != NULL)
    {
        string s = piece;
        // keeping only the pieces which actually have something in them
        if (s.find_first_not_of(" \t") != string::npos)
            parts.push_back(s);

        piece = strtok(NULL, ";");
    }

    delete[] buffer;
    return parts;
}

// breaks one command into tokens, the special symbols become their own tokens
vector<string> InputParsing(const string &input)
{
    vector<string> tokens;
    string tmptkn;
    bool inQuote = false;
    char quoteChar = '\0';

    for (size_t i = 0; i < input.length(); i++)
    {
        char c = input[i];

        if (c == '"' || c == '\'')
        {
            if (inQuote && c == quoteChar) // closing quote
            {
                inQuote = false;
                quoteChar = '\0';
                if (!tmptkn.empty())
                {
                    tokens.push_back(tmptkn);
                    tmptkn.clear();
                }
            }
            else if (!inQuote) // opening quote
            {
                inQuote = true;
                quoteChar = c;
            }
            else // the other kind of quote inside a quote, keep it as it is
            {
                tmptkn += c;
            }
        }
        else if (isspace((unsigned char)c) && !inQuote)
        {
            // spaces and tabs are separating the tokens, extra ones are ignored
            if (!tmptkn.empty())
            {
                tokens.push_back(tmptkn);
                tmptkn.clear();
            }
        }
        else if (!inQuote && (c == '|' || c == '<' || c == '>' || c == '&'))
        {
            if (!tmptkn.empty())
            {
                tokens.push_back(tmptkn);
                tmptkn.clear();
            }

            if (c == '>' && i + 1 < input.length() && input[i + 1] == '>')
            {
                tokens.push_back(">>"); // Handling  >>
                i++;
            }
            else
            {
                tokens.push_back(string(1, c));
            }
        }
        else
        {
            tmptkn += c;
        }
    }

    if (!tmptkn.empty())
        tokens.push_back(tmptkn);

    return tokens;
}
