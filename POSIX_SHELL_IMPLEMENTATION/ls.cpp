#include "functions.h"
#include <iostream>
#include <sys/stat.h>
#include <dirent.h>
#include <pwd.h>
#include <grp.h>
#include <unistd.h>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <iomanip>

using namespace std;

// builds the permission string like -rw-r--r--
static string getPermissions(struct stat &status)
{
    string perm;

    if (S_ISDIR(status.st_mode))
        perm += 'd';
    else if (S_ISLNK(status.st_mode))
        perm += 'l';
    else
        perm += '-';

    perm += ((status.st_mode & S_IRUSR) ? 'r' : '-');
    perm += ((status.st_mode & S_IWUSR) ? 'w' : '-');
    perm += ((status.st_mode & S_IXUSR) ? 'x' : '-');
    perm += ((status.st_mode & S_IRGRP) ? 'r' : '-');
    perm += ((status.st_mode & S_IWGRP) ? 'w' : '-');
    perm += ((status.st_mode & S_IXGRP) ? 'x' : '-');
    perm += ((status.st_mode & S_IROTH) ? 'r' : '-');
    perm += ((status.st_mode & S_IWOTH) ? 'w' : '-');
    perm += ((status.st_mode & S_IXOTH) ? 'x' : '-');

    return perm;
}

// prints one line of "ls -l" output, widths are passed so columns line up
static void PrintLongFormat(const string &path, const string &name,
                            int wLink, int wUser, int wGroup, int wSize)
{
    struct stat status;
    if (lstat(path.c_str(), &status) == -1)
    {
        perror("ls");
        return;
    }

    cout << getPermissions(status);

    cout << " " << setw(wLink) << status.st_nlink;

    struct passwd *pw = getpwuid(status.st_uid);
    struct group *gr = getgrgid(status.st_gid);
    string uname = pw ? pw->pw_name : to_string(status.st_uid);
    string gname = gr ? gr->gr_name : to_string(status.st_gid);

    cout << " " << setw(wUser) << left << uname << right;
    cout << " " << setw(wGroup) << left << gname << right;

    cout << " " << setw(wSize) << status.st_size;

    char timebuf[120];
    strftime(timebuf, sizeof(timebuf), "%b %e %H:%M", localtime(&status.st_mtime));
    cout << " " << timebuf;

    cout << " " << name << endl;
}

void ListDir(const string &path, bool show_all, bool long_format)
{
    DIR *dir = opendir(path.c_str());
    if (!dir)
    {
        perror(("ls: It cannot open " + path).c_str());
        return;
    }

    // first collecting all the names, then sorting them
    vector<string> names;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
        string name = entry->d_name;

        if (!show_all && name[0] == '.')
            continue;

        names.push_back(name);
    }
    closedir(dir);

    sort(names.begin(), names.end());

    if (!long_format)
    {
        for (size_t i = 0; i < names.size(); i++)
            cout << names[i] << endl;
        return;
    }

    // for -l we first go through everything once to find the total blocks
    // and how wide each column has to be
    long total = 0;
    int wLink = 1, wUser = 1, wGroup = 1, wSize = 1;

    for (size_t i = 0; i < names.size(); i++)
    {
        struct stat status;
        string fullpath = path + "/" + names[i];
        if (lstat(fullpath.c_str(), &status) == -1)
            continue;

        total += status.st_blocks / 2; // st_blocks is in 512 byte units, ls shows 1K

        struct passwd *pw = getpwuid(status.st_uid);
        struct group *gr = getgrgid(status.st_gid);
        string uname = pw ? pw->pw_name : to_string(status.st_uid);
        string gname = gr ? gr->gr_name : to_string(status.st_gid);

        if ((int)to_string(status.st_nlink).length() > wLink)
            wLink = to_string(status.st_nlink).length();
        if ((int)uname.length() > wUser)
            wUser = uname.length();
        if ((int)gname.length() > wGroup)
            wGroup = gname.length();
        if ((int)to_string(status.st_size).length() > wSize)
            wSize = to_string(status.st_size).length();
    }

    cout << "total " << total << endl;

    for (size_t i = 0; i < names.size(); i++)
    {
        string fullpath = path + "/" + names[i];
        PrintLongFormat(fullpath, names[i], wLink, wUser, wGroup, wSize);
    }
}

void lsCommand(const vector<string> &tokens)
{
    bool show_all = false;
    bool long_format = false;
    vector<string> paths;

    for (size_t i = 1; i < tokens.size(); i++)
    {
        if (tokens[i].length() > 1 && tokens[i][0] == '-')
        {
            bool badflag = false;
            for (char flag : tokens[i].substr(1))
            {
                if (flag == 'a')
                    show_all = true;
                else if (flag == 'l')
                    long_format = true;
                else
                {
                    cerr << "ls: It is invalid option -- '" << flag << "'" << endl;
                    badflag = true;
                    break;
                }
            }
            if (badflag)
                continue; // skipping this flag but still doing the rest
        }
        else
        {
            string resolved_path = tokens[i];
            if (tokens[i] == "~")
                resolved_path = MyHomeDir;
            else if (tokens[i].compare(0, 2, "~/") == 0)
                resolved_path = MyHomeDir + tokens[i].substr(1);
            paths.push_back(resolved_path);
        }
    }

    if (paths.empty())
        paths.push_back(".");

    for (size_t i = 0; i < paths.size(); i++)
    {
        struct stat status;
        if (stat(paths[i].c_str(), &status) == -1)
        {
            perror(("ls: It cannot access " + paths[i]).c_str());
            continue;
        }

        if (paths.size() > 1)
            cout << paths[i] << ":" << endl;

        if (S_ISDIR(status.st_mode))
        {
            ListDir(paths[i], show_all, long_format);
        }
        else
        {
            if (long_format)
            {
                // a single file so no need to calculate column widths
                PrintLongFormat(paths[i], paths[i], 1, 1, 1, 1);
            }
            else
            {
                cout << paths[i] << endl;
            }
        }

        if (i < paths.size() - 1)
            cout << endl;
    }
}
