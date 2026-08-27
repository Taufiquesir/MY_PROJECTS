# POSIX Shell

A Unix shell written from scratch in C++ — process control, pipelines, I/O
redirection, signals and job control, with no use of `system()`, `popen()` or
the C++ `filesystem` library.

```bash
make && ./myshell
```

```
<user@machine:~> ls -la | grep .cpp | wc -l
17
<user@machine:~> sleep 30 &
71301
<user@machine:~> jobs
[1] Running   sleep [71301]
```

### What it does

- **8 built-ins** — `cd`, `echo`, `pwd`, `ls`, `pinfo`, `search`, `history`, `jobs`
- **Pipelines of any length**, with redirection at any stage
- **Job control** — background jobs, Ctrl-C, Ctrl-Z, Ctrl-D
- **TAB completion** for filenames and for commands scanned from `$PATH`
- **Persistent history** across sessions, with arrow-key recall

### Things I am proud of

- **`ls -l` output is byte-identical to the system `ls`** — verified with `diff`,
  including block totals and column alignment
- **Pipelines never hang.** A reader only sees EOF once *every* write end is
  closed, so the parent closes all `2(n-1)` pipe descriptors after forking —
  the bug that makes `cat file | head -2` block forever
- **Async-signal-safe `SIGCHLD` handling.** The handler uses `write()` rather
  than `cout`, and drains children in a loop because signals are not queued
- **Redirection works on built-ins too**, which run in-process — the shell saves
  its own descriptors, flushes, then restores them
- **Zero warnings** under `-Wall -Wextra`

Full write-up: **[SHELL_BOOK.pdf](SHELL_BOOK.pdf)** — a 52-page implementation
study covering every module, system call and edge case.
Testing: **[TESTING_GUIDE.md](TESTING_GUIDE.md)** explains every test case.

| | |
|---|---|
| Language | C++11 |
| Source files | 17 `.cpp` + 1 header |
| Total lines | about 1841 |
| External library | GNU readline (`-lreadline`) |
| Compiler warnings | 0 with `-Wall -Wextra` |

> Built for Advanced Operating Systems, Assignment 2 (Monsoon 2025).

---

## Table of Contents

1. [Compilation and Running](#1-compilation-and-running)
2. [File Structure](#2-file-structure)
3. [How a Command is Executed](#3-how-a-command-is-executed)
4. [Parsing](#4-parsing)
5. [Requirement wise Implementation](#5-requirement-wise-implementation)
6. [Libraries and Headers Used](#6-libraries-and-headers-used)
7. [System Calls Used](#7-system-calls-used)
8. [Important Problems Faced and How They Were Solved](#8-important-problems-faced-and-how-they-were-solved)
9. [Error Handling](#9-error-handling)
10. [Testing](#10-testing)
11. [Assignment Rules Followed](#11-assignment-rules-followed)
12. [Known Limitations](#12-known-limitations)

---

## 1. Compilation and Running

```bash
make clean      # remove the old build
make            # compile the shell
./myshell       # start the shell
```

The makefile uses:

```makefile
CXXFLAGS = -Wall -Wextra -std=c++11 -g
LDFLAGS  = -lreadline
```

Every object file rule depends on `functions.h`, so if the header is changed
then all the files which include it are compiled again. `clean` is marked as
`.PHONY` so that it runs even if a file named `clean` is present in the folder.

To come out of the shell you can type `exit` or press **Ctrl-D**.

---

## 2. File Structure

```
├── README.md
├── makefile
├── functions.h
├── main.cpp            parser.cpp        executor.cpp
├── display.cpp         cd.cpp            echo.cpp          pwd.cpp
├── ls.cpp              pinfo.cpp         search.cpp
├── history.cpp         jobs.cpp
├── system.cpp          pipe.cpp          redirection.cpp
├── signal.cpp          autocomplete.cpp
```

| File | What it does | Requirement | Lines |
|------|--------------|-------------|-------|
| `functions.h` | All declarations, the `Redirection` and `Job` structs, the two globals | — | 79 |
| `main.cpp` | The main loop — read a line, save it in history, split it and run it | — | 81 |
| `parser.cpp` | `SplitBySemicolon` (uses **strtok**) and `InputParsing` (the tokenizer) | — | 103 |
| `executor.cpp` | Decides built-in or system command, and gives redirection to built-ins | — | 137 |
| `display.cpp` | Builds the prompt | 1 | 48 |
| `cd.cpp` | `cd` command | 2 | 61 |
| `echo.cpp` | `echo` command | 2 | 16 |
| `pwd.cpp` | `pwd` command | 2 | 18 |
| `ls.cpp` | `ls` with `-a` and `-l` | 3 | 215 |
| `system.cpp` | Foreground and background system commands | 4 | 128 |
| `pinfo.cpp` | `pinfo` command | 5 | 139 |
| `search.cpp` | `search` command | 6 | 75 |
| `redirection.cpp` | Parsing of `<`, `>` and `>>` | 7 | 62 |
| `pipe.cpp` | Pipelines and redirection inside them | 8, 9 | 192 |
| `signal.cpp` | Ctrl+C, Ctrl+Z and background cleaning | 10 | 121 |
| `jobs.cpp` | Keeps the list of background and stopped jobs | supports 10 | 62 |
| `autocomplete.cpp` | TAB completion for commands and files | 11 | 172 |
| `history.cpp` | `history` command and the arrow keys | 12 | 132 |

### The two global variables

Only two globals are used. They are declared as `extern` in `functions.h` and
defined once in `main.cpp`:

```cpp
std::string MyHomeDir;  // the directory from which the shell was started, shown as ~
std::string PrevDir;    // needed for "cd -"
```

`PrevDir` is the shell's own variable and **not** the environment variable
`OLDPWD`, because the assignment does not allow `PWD` and `OLDPWD`. It is set
to `MyHomeDir` at the start so that `cd -` works even in a fresh shell instead
of giving an error.

---

## 3. How a Command is Executed

```
readline()  ->  add_to_history()  ->  SplitBySemicolon()  ->  InputParsing()
                                                                    |
                                                                    v
                                                               Runcmd()
                                                          (remove & , check |)
                                                                    |
                                    +-------------------------------+
                                    |                               |
                              is it built-in?                   otherwise
                                    |                               |
                                    v                               v
                             runBuiltin()                   fork() + execvp()
                          (same process, so             (new process, shell
                           cd can change the             waits only if it is
                           shell's directory)            in the foreground)
                                                                    |
                                            if there is a '|' then pipe.cpp
                                            takes over before all of this
```

The order of these steps has been kept like this on purpose:

- **History is saved before parsing.** So even a wrong command is remembered,
  which is how the real shells also behave.
- **The `&` is removed before the pipeline is checked.** So `a | b &` is
  understood as a background pipeline and not as a pipeline whose last word is
  `&`.

---

## 4. Parsing

Parsing is done in two stages, both in `parser.cpp`.

### Stage 1 — splitting on `;` with strtok

The assignment says that `strtok` must be used. The problem with `strtok` is
that it puts `\0` inside the string which is given to it, so it cannot be used
directly on a `std::string`. That is why a copy is made first:

```cpp
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
```

`strtok(NULL, ";")` continues from where the last call stopped, because
`strtok` keeps a static pointer inside it. This is also the reason why
`strtok` is not reentrant. The whitespace check throws away the empty pieces,
so `ls ;; pwd` and a `;` at the end do not create empty commands.

### Stage 2 — the tokenizer

`strtok` cannot be used for this stage because it cannot tell `>` from `>>`
and it cannot understand quotes. So a character by character scanner is
written, which keeps two things in mind:

```cpp
bool inQuote = false;
char quoteChar = '\0';
```

For every character there are four cases:

1. **A quote character** — it opens a quote, or closes it if it is the same
   quote, or it is kept as a normal character if it is the other type of quote
   inside a quote. This is how `"abc 'ac' abc"` keeps its inner single quotes.
2. **Space or tab outside a quote** — the current token ends here. Many spaces
   together do not create empty tokens, because an empty token is never pushed.
   This is what handles "the user can type the command anywhere with random
   spaces and tabs".
3. **A special symbol outside a quote** (`|`, `<`, `>`, `&`) — the current
   token is pushed first, and then the symbol is pushed as a **separate
   token**. One character is looked ahead to join `>>`.
4. **Anything else** — added to the current token.

Making the symbols separate tokens is the most useful decision here. Because
of it, `ls>out.txt` written without spaces gives exactly the same tokens as
`ls > out.txt`, and after this stage every other part of the code can simply
compare full tokens with `==`.

---

## 5. Requirement wise Implementation

### Requirement 1 — Display (`display.cpp`)

The prompt is `<username@systemname:current_directory>`. Nothing is hard
coded, all three parts are found at runtime:

| Part | Function used | Note |
|------|---------------|------|
| Username | `getpwuid(getuid())` | reads the passwd database, `pw_name` is the login name |
| System name | `gethostname()` | return value is checked and the buffer is ended with `\0` by hand, because it is not guaranteed when the name is too long |
| Directory | `getcwd()` | taken freshly every time the prompt is printed |

The directory from which the shell was started is shown as `~`:

```cpp
if (!MyHomeDir.empty() && currDir.compare(0, MyHomeDir.length(), MyHomeDir) == 0)
{
    // it must be the home itself or something inside it, not just a name
    // which starts with the same letters
    if (currDir.length() == MyHomeDir.length() || currDir[MyHomeDir.length()] == '/')
        currDir.replace(0, MyHomeDir.length(), "~");
}
```

The inner `if` is important. Without it, if home is `/home/user/shell` then
the directory `/home/user/shell_backup` would wrongly become `~_backup`. By
checking that the next character is a `/` (or that the string ends there), the
match happens only at a proper directory boundary.

### Requirement 2 — cd, echo, pwd

These are built-ins and `execvp` is **not** used for them.

#### cd (`cd.cpp`)

The argument is first turned into a target path and then only one `chdir` call
is made. The old directory is taken **before** moving, and it is saved into
`PrevDir` only if the `chdir` was successful, so a failed `cd` does not spoil
the target of `cd -`.

| Form | Goes to |
|------|---------|
| `cd` | `MyHomeDir` (no argument means home) |
| `cd ~` | `MyHomeDir` |
| `cd ~/sub` | `MyHomeDir + "/sub"` |
| `cd -` | `PrevDir`, and the path is printed like bash does |
| `cd .` | given directly to `chdir` |
| `cd ..` | given directly to `chdir` |
| more than 1 argument | prints `Invalid arguments` |

`.` and `..` are **not** handled separately on purpose. The kernel already
knows them as real directory entries, so `chdir("..")` is correct, and writing
our own parent-path logic would only create bugs with symlinks and with root.

#### echo (`echo.cpp`)

```cpp
for (size_t i = 1; i < tokens.size(); i++)
{
    if (i > 1)
        cout << " ";
    cout << tokens[i];
}
cout << endl;
```

It prints the **tokens**, with a single space in between. This looks very
simple but it is the important point: an earlier version was printing the
original input line after removing the word `echo`, and that was wrong,
because then the redirection and pipe symbols were printed as normal text. So
`echo hello > output.txt` was printing `hello > output.txt` and no file was
made. Printing the tokens (after the redirection has already been removed from
them) is what makes `echo` work correctly with redirection and pipes.

The quotes are already removed by the tokenizer, so tabs and many spaces
inside a quote are kept and the spaces outside are made into one.

#### pwd (`pwd.cpp`)

One `getcwd` call, the `NULL` return is checked and reported using `perror`.
It asks the kernel every time instead of reading the `PWD` environment
variable, which is anyway not allowed and can also become old after a `chdir`
through a symlink.

### Requirement 3 — ls (`ls.cpp`)

The biggest built-in. `execvp` is not used anywhere in it. `opendir`,
`readdir`, `stat` and `lstat` are used instead.

**Arguments:** one loop separates the flags from the paths, so the flags and
the directory names can come in any order and mixed together, like
`ls -l dir1 -a dir2`. The flag letters are read one by one from the string
after the `-`, so `-la`, `-al` and `-a -l` automatically become the same. If a
wrong flag comes then it is reported and the loop does `continue`, so the
correct part of the command still runs.

**Long format is done in two passes.** The real `ls -l` puts its number
columns to the right side, and how wide they should be can be known only after
seeing all the entries. So:

```cpp
// pass 1 - find the total and the widths
total += status.st_blocks / 2;   // st_blocks counts 512 byte units, ls shows 1K
if ((int)to_string(status.st_size).length() > wSize)
    wSize = to_string(status.st_size).length();

cout << "total " << total << endl;

// pass 2 - print using the widths from pass 1
cout << " " << setw(wSize) << status.st_size;
```

The `/ 2` is a small detail which is easy to miss — `st_blocks` is counted in
512 byte units but `ls` shows 1K blocks.

**Reading the information:**

- Type and permissions come from `st_mode`, by checking it against `S_IRUSR`,
  `S_IWGRP`, `S_IXOTH` and so on. `S_ISDIR` and `S_ISLNK` decide the first
  character.
- Owner and group come from `getpwuid(st_uid)` and `getgrgid(st_gid)`. If the
  lookup fails then the number itself is printed.
- Time comes from `localtime` and `strftime` with `"%b %e %H:%M"`. `%e` puts a
  space before a single digit date and `%d` puts a zero — the real `ls` uses
  `%e`.

The names are put in a `vector` and **sorted**, because `readdir` gives them in
the order of the directory which is not fixed. `lstat` is used instead of
`stat` so that a symlink is shown as a symlink.

> The output was compared with the real `ls -l` of the system on the same
> directory and it is exactly the same, including the `total` line and the
> spacing of all the columns.

### Requirement 4 — System commands (`system.cpp`)

Any command which is not a built-in is run like this:

```cpp
pid_t pid = fork();

if (pid == 0)                     // CHILD
{
    setpgid(0, 0);                // its own process group
    signal(SIGINT,  SIG_DFL);     // remove the shell's handlers
    signal(SIGTSTP, SIG_DFL);
    // ... redirection with dup2 ...
    execvp(cargs[0], cargs.data());
    perror(cmdtokens[0].c_str()); // reached only if execvp failed
    _exit(1);
}
else if (pid > 0)                 // PARENT
{
    setpgid(pid, pid);            // done here also, to remove the race
    ...
}
```

**Why `fork` returns two times.** `fork` makes a copy of the process. It
returns `0` in the child and the child's pid in the parent, and that is the
only way both of them know who they are. The memory is copy-on-write, so it is
not really copied until somebody writes into it.

**Why `setpgid` is written in both the places.** Whichever process runs first
should be able to make the group. If only the child had called it, then the
parent might try to signal a group which is not yet made. Calling it on both
sides is the normal way, and the second call does no harm because it sets the
same value.

**Why `_exit` and not `exit`.** `exit` runs the `atexit` functions and flushes
the stdio buffers. In a forked child those buffers are a **copy** of the
parent's buffers, so flushing them would print the parent's pending output
once more. `_exit` finishes at once without touching them.

**Why `execvp`.** The `v` means the arguments are given as a vector
(`char* const[]` ending with `NULL`) and the `p` means that `PATH` is searched.
If it succeeds it never comes back, because the whole process image is
replaced. So any line written after it is the failure case.

**Foreground and background:**

| Mode | What the parent does | What is printed |
|------|---------------------|-----------------|
| Foreground | `waitpid(pid, &status, WUNTRACED)` after storing the pid as the foreground one | nothing |
| Background `&` | does not wait, only adds the job to the list | the PID of the new process |

`WUNTRACED` is what makes Ctrl-Z work. Without it `waitpid` comes back only
when the child **finishes**, so a stopped child would hang the shell forever.
With it, `waitpid` also returns when the child is stopped, and then
`WIFSTOPPED(status)` tells us which of the two happened, so the job can be
kept as a stopped job.

The input file is opened once in the parent only to check that it exists,
before `fork` is called. This way the "input file does not exist" error is
given by the shell itself and not by a child which is already made.

### Requirement 5 — pinfo (`pinfo.cpp`)

`/proc` is read directly. `popen` is not allowed and is also not needed.
`/proc/<pid>/stat` is one line having around 52 fields separated by spaces.

**The problem with the name field.** The second field is the name of the
program **inside brackets**, and this name can itself have spaces in it (for
example Firefox has a process named `(Web Content)`). So if the line is simply
split on spaces then every field after it moves and all the values become
wrong. The solution is to start reading after the **last** `)`:

```cpp
size_t bracket = line.rfind(')');
istringstream iss(line.substr(bracket + 1));

// fields[0] is now the state, which is field 3 in the original file,
// so field number N of the file is fields[N - 3] here
string state = fields[0];                 // field 3
long   pgrp  = stol(fields[2]);           // field 5
long   tpgid = stol(fields[4]);           // field 7
unsigned long vsize = stoul(fields[20]);  // field 23
```

`rfind` is used and not `find`, because the name itself can also contain a
bracket.

**Finding the foreground `+`:**

```cpp
if (tpgid != -1 && tpgid == pgrp)
    formatted_state += "+";
```

`tpgid` is the process group which is holding the terminal right now, and
`pgrp` is the group of this process. If both are same then this process is the
foreground one, which is exactly what the `+` in `R+` or `S+` means.

**Executable path.** `/proc/<pid>/exe` is a symlink to the real program and it
is read with `readlink`. But it fails for the processes of other users. So if
it fails then `/proc/<pid>/cmdline` is read instead. In that file the
arguments are separated by **null characters**, that is why
`getline(cmd_file, cmdline, '\0')` is used to take only the first one. Before
adding this, `pinfo 1` was printing `N/A` and now it prints `/sbin/init`.

| Code | Meaning |
|------|---------|
| `R` / `R+` | Running |
| `S` / `S+` | Sleeping in an interruptible wait |
| `Z` | Zombie |
| `T` | Stopped by a signal |

### Requirement 6 — search (`search.cpp`)

Goes into the current directory recursively and prints `True` or `False`. It
finds both files and folders, because the name is compared before checking the
type.

```cpp
if (name == "." || name == "..") continue;   // else it will never stop

if (name == target) { closedir(dir); return true; }

struct stat statbuff;
if (lstat(fullPath.c_str(), &statbuff) == 0)
    if (S_ISDIR(statbuff.st_mode))
        if (RecurSearch(fullPath, target)) { closedir(dir); return true; }
```

**`lstat` and not `stat`.** `stat` follows the symlinks, so a link which
points to a directory above it makes a circle and the recursion never
finishes. `lstat` gives the information of the link itself, which is not a
directory, so it is not entered. An earlier version had written
`S_ISDIR(m) && !S_ISLNK(m)` along with `stat`, which is useless code, because
`stat` has already followed the link and can never say `S_ISLNK`.

`closedir` is called on **every** return path including the early `true`
returns, otherwise the open directories keep increasing and on a deep tree the
file descriptors will finish.

### Requirement 7 — I/O redirection (`redirection.cpp`)

It takes the token vector, finds the redirection, and changes the vector so
that only the actual command and its arguments are left. Since the tokenizer
has already made the symbols into separate tokens, this is a simple loop — on
seeing `<`, `>` or `>>` the **next** token is taken as the file name and
skipped, and anything else is kept.

| Operator | Flags given to `open()` | Meaning |
|----------|------------------------|---------|
| `<` | `O_RDONLY` | error if the file is not there |
| `>` | `O_WRONLY \| O_CREAT \| O_TRUNC`, `0644` | make it or empty it |
| `>>` | `O_WRONLY \| O_CREAT \| O_APPEND`, `0644` | make it or add at the end |

If the file name is missing after an operator then `error = true` is set in the
struct, and every place which calls this function checks it. So `echo x >`
gives a syntax error instead of being ignored silently.

**About 0644 and umask.** The mode given to `open` is only a request, the
kernel applies `mode & ~umask`. With the usual umask `022` the file really
gets `0644`, and this was checked with the `stat` command. With a stricter
umask the file will get less permissions, which is the correct POSIX behaviour
and not a mistake.

**Redirection for built-in commands** is done in `executor.cpp`. A built-in
runs inside the shell itself, so its stdin/stdout are changed for some time and
then put back:

```cpp
int savedIn  = dup(STDIN_FILENO);    // keep a spare handle of the terminal
int savedOut = dup(STDOUT_FILENO);

// ... open the file and dup2 it on fd 0 or 1 ...

if (ok) runBuiltin(cmdtokens);

cout.flush();                        // very important, explained below
dup2(savedIn,  STDIN_FILENO);        // put everything back
dup2(savedOut, STDOUT_FILENO);
close(savedIn);
close(savedOut);
```

> **The flush problem.** `cout` keeps the data in its own buffer. Without the
> `cout.flush()`, the output of the built-in would still be sitting inside the
> C++ buffer when `dup2` puts fd 1 back, and then it would go to the
> **terminal** instead of the file, and the file would remain empty. This
> mistake is not seen in small tests because sometimes the buffer flushes on
> its own.

`dup` gives the smallest free descriptor number pointing to the same file, and
`dup2(old, new)` forces `new` to become a copy of `old` after closing whatever
`new` was before. This closing and reusing happens in one step, and that is why
`dup2` is used and not `close` followed by `dup`.

### Requirements 8 and 9 — Pipelines with redirection (`pipe.cpp`)

Any number of pipes is supported. For *n* commands, *n−1* pipes and *n*
children are made. The pipes are kept in one flat array:

```cpp
vector<int> pipefds(2 * pipecount);
for (int i = 0; i < pipecount; i++)
    pipe(&pipefds[i * 2]);
```

`pipe()` fills two numbers — index `0` is the **read** end and index `1` is the
**write** end. So pipe number *i* is at `pipefds[i*2]` and `pipefds[i*2+1]`,
and command number *i* reads from `pipefds[(i-1)*2]` and writes into
`pipefds[i*2+1]`.

**A file is more important than the pipe:**

```cpp
if (!redirs[i].inputfile.empty()) { /* open the file, dup2 on stdin */ }
else if (i > 0)  dup2(pipefds[(i - 1) * 2], STDIN_FILENO);

if (!redirs[i].outputfile.empty()) { /* open the file, dup2 on stdout */ }
else if (i < n - 1) dup2(pipefds[i * 2 + 1], STDOUT_FILENO);
```

This `else if` is what makes a command like `ls | grep .txt > out.txt` work
correctly — the `>` file wins over the pipe.

> **The most important point — closing the descriptors.** Every child gets a
> copy of **all** the pipe descriptors and the parent also has them. A reading
> process gets EOF only when the **last** writing end is closed. So if even one
> process keeps a write end open, the reader waits forever and the whole
> pipeline hangs. That is why every child closes all the `2 × pipecount`
> descriptors after doing its `dup2`, and the parent also closes all of them
> after making all the children. This was checked with
> `cat /etc/passwd | head -2`, which finishes properly.

The redirection of all the commands is parsed **before** any `fork` happens,
so if there is a syntax mistake then the shell reports it one time instead of
several half made children reporting it.

Built-ins are also allowed inside a pipeline — the child calls `runBuiltin`,
flushes and does `_exit`. Because of this, `cd sub | wc` changes the directory
of a child which dies immediately, so the shell's own directory does not
change. Bash also behaves in the same way.

### Requirement 10 — Signals (`signal.cpp`)

| Key | Signal | What happens |
|-----|--------|--------------|
| Ctrl-C | `SIGINT` | sent only to the foreground job; if there is none, the shell is not affected |
| Ctrl-Z | `SIGTSTP` | stops the foreground job and puts it in the job list |
| Ctrl-D | EOF (not a signal) | `readline` returns `NULL`, the loop breaks and the shell exits |
| — | `SIGCHLD` | reaps the finished background children and prints a message |

Ctrl-D is not a signal at all. It is the terminal driver sending end-of-file,
which comes to us as a `NULL` return from `readline`.

**Why `sigaction` and not `signal`.** `sigaction` has fixed and portable
behaviour — the handler stays installed after it runs once, and `SA_RESTART`
makes the interrupted system calls start again instead of failing with
`EINTR`. The old `signal()` behaves differently on different systems.
`SA_NOCLDSTOP` is given for SIGCHLD so that a child which only **stops** does
not call the reaping handler.

**Async-signal-safety.** Only some fixed functions are allowed inside a signal
handler. `cout` is not one of them, because it is not reentrant and a signal
coming in the middle of a `cout` in the main code can spoil the stream. So the
SIGCHLD handler uses the plain `write` system call:

```cpp
string msg = "\n" + name + " with pid " + to_string(pid) + " exited\n";
ssize_t r = write(STDOUT_FILENO, msg.c_str(), msg.length());
(void)r;   // to remove the unused-result warning
```

> **The SIGCHLD race — the most difficult bug which was fixed.**
> The handler calls `waitpid(-1, ...)` which means **any** child. When a
> foreground job finishes, SIGCHLD can come and reap it **before** the
> `waitpid(pid, ...)` of `system.cpp` runs. Then that `waitpid` fails with
> `ECHILD` because the child is already gone, and the exit status as well as
> the `WIFSTOPPED` information is lost.
>
> The solution is a `volatile sig_atomic_t waitingFg` flag which is set
> whenever a foreground pid is stored. The handler returns immediately while
> this flag is set, and leaves the foreground children for the proper
> `waitpid`. `sig_atomic_t` is the type which is guaranteed to be read and
> written in one step which cannot be interrupted, and `volatile` stops the
> compiler from keeping it in a register.

The children set `SIGINT` and `SIGTSTP` back to `SIG_DFL` after the fork,
because the handlers are inherited through `fork`. Without this the child
would also inherit the shell's forwarding handler and would ignore the signal
instead of dying. The children are also put into their own process group, so
that the signals coming from the terminal reach the job and not the shell.

**Zombies.** A finished child stays in the process table as a zombie until its
parent reads its exit status. Reaping it in the SIGCHLD handler is what stops
a long running shell from collecting zombies, which can be seen as `Z` in
`pinfo`.

The `jobs` command shows all the background and stopped processes.

### Requirement 11 — Autocomplete (`autocomplete.cpp`)

Made using readline's completion system (ncurses is not used). Which generator
to call is decided from the position of the cursor:

```cpp
static char **customCompletion(const char *text, int start, int)
{
    rl_attempted_completion_over = 1;   // stop readline's own file completion

    bool isFirstWord = true;
    for (int i = 0; i < start; i++)
        if (rl_line_buffer[i] != ' ' && rl_line_buffer[i] != '\t')
            isFirstWord = false;

    if (isFirstWord)
        return rl_completion_matches(text, cmdGenerator);   // commands
    else
        return rl_completion_matches(text, filenameGntr);   // files
}
```

**How a generator works.** readline calls it again and again. `state == 0`
means a new completion has started, and that is when the list of matches is
made. Every call after that returns one more match, and `NULL` when they are
finished. The returned strings must be made with `malloc` (so `strdup` is
used) because readline takes them and frees them itself.

**The command list is made from PATH.** Our built-in names plus every program
found in every directory of `PATH`. `PATH` is split on `:`, every directory is
opened with `opendir`, and only those entries are kept which pass
`access(full, X_OK)`. Then the list is sorted and the duplicates are removed
with `unique`, because one program can be present in more than one directory.
The whole scan is done only once and kept behind a `commandsLoaded` flag.

> Checked with the examples of the assignment: `ca` + TAB + TAB shows
> `capsh cat catman ...` and `cat alp` + TAB completes to `alpha.txt`.
> One thing to note — on a machine which also has `ec2metadata`, pressing TAB
> after `ec` fills only `ec`, because there are two matches and stopping at the
> common part is the correct behaviour. Pressing TAB again shows both.

### Requirement 12 — history (`history.cpp`)

A ring of 20 commands kept in a `vector<string>`, and saved in
`~/.myshell_history` so that it works across all the sessions.

```cpp
CmdHistory.push_back(trimmedCmd);
while ((int)CmdHistory.size() > MAX_HISTORY)
    CmdHistory.erase(CmdHistory.begin());   // the oldest one goes out

add_history(trimmedCmd.c_str());   // give it to readline for the arrow keys
save_history();                    // write the file after every command
```

| Rule | Value | Reason |
|------|-------|--------|
| Commands stored | 20 | given in the assignment, oldest is overwritten |
| Shown by default | 10 | given in the assignment |
| `history <num>` | brought down to 20 if more | cannot show more than what is stored |
| Same command twice | **kept** | the example in the pdf shows `ls, cd, ls`, so removing the duplicates would break it |
| `history` itself | **saved** | the example in the pdf shows `history` inside its own output |

The last two rows were mistakes in the earlier version. Removing the
duplicates and not saving `history` looked neater but it does not match the
examples given in the assignment pdf.

The file is written after **every** command and not at the exit, so the history
is safe even if the shell is closed by a `kill` or crashes.

**Arrow keys** are not written by hand. At the start `LoadHistory` puts the
saved commands into readline using `add_history`, and after that readline
itself gives the up/down movement, the editing of the command which comes up,
and the behaviour of staying on the same command at the two ends.

---

## 6. Libraries and Headers Used

### External library — GNU readline

This is the only library which is not standard, and it is linked with
`-lreadline`. It gives the line editing, the movement of the cursor, the
history with arrow keys and the framework for TAB completion. It is used
because the assignment wants arrow keys and TAB completion but does not allow
ncurses, and writing a raw mode terminal editor by hand would be much more
code and much more risky.

| Symbol | Work |
|--------|------|
| `readline()` | shows the prompt, lets the user edit, returns the line (`NULL` on Ctrl-D) |
| `add_history()` | puts a line into readline's own history |
| `clear_history()` | clears it before loading the saved file again |
| `rl_attempted_completion_function` | the hook for our own completion |
| `rl_completion_matches()` | runs a generator and finds the common prefix |
| `rl_line_buffer` | the text of the current line |
| `rl_bind_key()` | binds the TAB key |

### POSIX and C headers

| Header | What is taken from it | Used in |
|--------|----------------------|---------|
| `unistd.h` | `fork execvp dup dup2 pipe chdir getcwd access read write _exit setpgid getpid getuid readlink` | almost all files |
| `sys/wait.h` | `waitpid`, `WIFSTOPPED`, `WNOHANG`, `WUNTRACED` | system, pipe, signal |
| `fcntl.h` | `open` and the `O_*` flags | functions.h |
| `dirent.h` | `opendir readdir closedir`, `struct dirent` | ls, search, autocomplete |
| `sys/stat.h` | `stat lstat`, `S_IS*`, the permission bits | ls, search, autocomplete |
| `pwd.h` | `getpwuid`, `struct passwd` | display, ls, history |
| `grp.h` | `getgrgid`, `struct group` | ls |
| `csignal` | `sigaction`, `kill`, `sig_atomic_t` | signal, system, pipe |
| `cstring` | `strtok strcpy strlen strdup` | parser, autocomplete |
| `ctime` | `localtime strftime` | ls |

### C++ standard library

`vector` and `string` are used everywhere, so there is no manual `malloc` or
`free` except the one buffer needed for `strtok`. `fstream` is used for the
history file, `sstream` for splitting the `/proc` fields, `iomanip` for `setw`
to align the `ls -l` columns, and `algorithm` for `sort` and `unique`.

---

## 7. System Calls Used

| Call | Purpose | Important point |
|------|---------|-----------------|
| `fork` | make a new process | returns twice — 0 in child, pid in parent |
| `execvp` | replace the process image | searches PATH, never returns if it succeeds |
| `waitpid` | wait for a child / reap it | `WUNTRACED` also returns on stop, `WNOHANG` does not block |
| `pipe` | make a one way channel | `fd[0]` read, `fd[1]` write |
| `dup` / `dup2` | copy a descriptor | `dup2` closes the target first, in one step |
| `open` / `close` | file descriptors | mode `0644` is filtered by umask |
| `chdir` / `getcwd` | working directory | it is a per-process kernel state |
| `stat` / `lstat` | information about a file | `lstat` does not follow symlinks |
| `readlink` | read where a symlink points | used on `/proc/pid/exe`, does not add `\0` |
| `opendir` / `readdir` | read a directory | the order is not fixed, so sorting is needed |
| `sigaction` | install a signal handler | portable, gives `SA_RESTART` and `SA_NOCLDSTOP` |
| `kill` | send a signal | used to forward SIGINT and SIGTSTP |
| `setpgid` | set the process group | called in both parent and child |
| `write` | plain output | safe inside a signal handler, `cout` is not |
| `access` | check the permission | `X_OK` is used to pick the programs from PATH |
| `getpwuid` / `getgrgid` | uid/gid to name | used in the prompt and in `ls -l` |
| `gethostname` | name of the machine | the buffer has to be ended by hand |

---

## 8. Important Problems Faced and How They Were Solved

| Problem | What was going wrong | How it was solved |
|---------|---------------------|-------------------|
| SIGCHLD taking away the foreground child | exit status was lost, `WIFSTOPPED` was never seen | a `volatile sig_atomic_t waitingFg` flag stops the handler |
| Pipe write ends not closed | the pipeline hangs forever, the reader never gets EOF | close all `2×(n−1)` descriptors in every child and in the parent |
| `cout` buffer and `dup2` | a redirected built-in was making an empty file | `cout.flush()` before putting the descriptors back |
| `exit` inside a forked child | the parent's output was printed two times | use `_exit`, it does not flush the stdio |
| Signal handlers going into the child | the child was ignoring Ctrl-C instead of dying | set them to `SIG_DFL` after fork, and use a separate process group |
| `echo` printing the raw line | `echo hi > f.txt` was printing `hi > f.txt` and making no file | print the tokens instead of the original input |
| `/proc/stat` split on spaces | a process named `(Web Content)` gave wrong values | start reading after the last `)` |
| `stat` in the recursion of search | a symlink circle could make it run forever | use `lstat` |
| `readdir` order in `ls` | the output was not matching the real `ls` | put the names in a vector and `sort` them |

---

## 9. Error Handling

- Wrong commands, wrong flags and wrong PIDs all print a proper message.
- `perror` is used so that the real reason from the system is also shown.
- Missing input files, missing file names after a redirection symbol, and an
  empty command before or after a `|` are all checked.
- `cd` with more than one argument prints `Invalid arguments`.
- The return values of `getcwd`, `gethostname`, `fork`, `pipe`, `open`,
  `opendir` and `sigaction` are checked.
- The argument of `pinfo` is checked to be a number before `stoi` is called, so
  that it does not throw.
- `history` checks that the whole argument is a number, so `history 12abc` is
  rejected.
- Empty lines and lines having only spaces or tabs are ignored.
- No segmentation fault was seen in any of the tests.

---

## 10. Testing

The shell was tested for all the requirements. Some of the important checks:

```bash
# echo with redirection and pipe
echo hello > out.txt          # the file gets "hello"
echo hello | wc -c            # gives 6
echo "abc 'ac' abc"           # inner quotes are kept

# ls compared with the real ls
ls -l                         # output is exactly same as the system ls -l
ls -l dir1 -a dir2            # flags and directories in any order
ls -la dir1 dir2

# redirection
sort < un.txt > s.txt
echo x >> out.txt
cat < missing.txt             # gives a proper error

# pipelines
cat /etc/passwd | cut -d: -f1 | sort | head -3
cat /etc/passwd | head -2     # finishes properly, does not hang
ls | grep ".txt" > o.txt ; cat < in.txt | wc -l > lines.txt

# background and pinfo
sleep 100 &                   # prints the pid
jobs
pinfo 1                       # prints /sbin/init

# signals (tested on a real terminal)
sleep 50   then Ctrl-C        # only the sleep dies, the shell stays
sleep 50   then Ctrl-Z        # goes to the job list as stopped
Ctrl-C with nothing running   # the shell is not affected
Ctrl-D                        # the shell exits

# history examples of the pdf
ls ; cd ; ls ; history        # matches the pdf exactly
history 3
```

---

## 11. Assignment Rules Followed

- Written only in **C++**.
- **`strtok`** is used for splitting the command list on `;`.
- **`popen`, `pclose` and `system()` are not used** anywhere.
- The **`filesystem`** library is not used.
- **ncurses** is not used.
- The environment variables **`PWD` and `OLDPWD` are not used** — the previous
  directory is kept in the shell's own variable.
- **`execvp` is not used** for `cd`, `echo`, `pwd`, `ls` and `pinfo`. It comes
  only in `system.cpp` and `pipe.cpp` for the other commands.
- Redirection and piping are implemented for the internal commands also.
- The output file is made with permission **0644**.
- The code **compiles without any warning** with `-Wall -Wextra`.
- Every command is in a separate file (modular).
- The prompt does not have the user name or the system name hard coded.

---

## 12. Known Limitations

These were left out on purpose, they are outside what the assignment asked:

- `fg` and `bg` are not there. A stopped job can be seen with `jobs` but it
  cannot be started again. To add it, `bg` would send `SIGCONT` to the group of
  the job, and `fg` would send `SIGCONT`, give the terminal to that group using
  `tcsetpgrp`, wait for it with `WUNTRACED`, and then take the terminal back.
- No variable expansion (`$HOME`) and no wildcards (`*.txt`).
- Redirection operators like `2>`, `&>` and `2>&1` are not done, which the
  assignment says is not needed.
- The quotes are removed but escape characters like `\n` are not handled, which
  the assignment also says is not needed.
- The history file is not locked, so if two shells are running together then
  one can overwrite the history of the other.
