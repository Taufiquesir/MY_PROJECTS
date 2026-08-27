# POSIX Shell — Complete Testing Guide

**Advanced Operating Systems, Assignment 2**

Every command in this guide is explained: what it does, why you are running it,
and what you should see. Work through the sections in order.

---

## How to use this file

| Symbol | Meaning |
|--------|---------|
| `$` | type it in your **normal** terminal (bash) |
| `>` | type it **inside my shell** (after the prompt appears) |
| `->` | the output you should get |

Anything marked **[MANUAL]** cannot be tested by piping input — you must type it
by hand on a real terminal. These are the signal and TAB tests.

> **Note:** the reference outputs below were taken on my machine, so the
> username, hostname, PIDs, file sizes, dates and memory numbers will be
> **different** on yours. The *shape* of the output is what should match, not
> those exact values.

---

## Contents

| # | Section | Requirement |
|---|---------|-------------|
| 0 | [Build and Setup](#section-0--build-and-setup) | — |
| 1 | [Prompt](#section-1--prompt) | 1 |
| 2 | [cd, echo, pwd](#section-2--cd-echo-pwd) | 2 |
| 3 | [ls](#section-3--ls) | 3 |
| 4 | [System commands, foreground and background](#section-4--system-commands-foreground-and-background) | 4 |
| 5 | [pinfo](#section-5--pinfo) | 5 |
| 6 | [search](#section-6--search) | 6 |
| 7 | [I/O redirection](#section-7--io-redirection) | 7 |
| 8 | [Pipelines](#section-8--pipelines) | 8 |
| 9 | [Redirection inside pipelines](#section-9--redirection-inside-pipelines) | 9 |
| 10 | [Signals](#section-10--signals-manual) **[MANUAL]** | 10 |
| 11 | [Autocomplete](#section-11--autocomplete-manual) **[MANUAL]** | 11 |
| 12 | [history](#section-12--history) | 12 |
| 13 | [Mixed and tricky cases](#section-13--mixed-and-tricky-cases) | — |
| 14 | [Crash test](#section-14--crash-test) | — |
| 15 | [Rules check](#section-15--rules-check) | — |
| 16 | [Before you submit](#section-16--before-you-submit) | — |
| ⭐ | [The tests that matter most](#the-tests-that-matter-most) | — |
| ✅ | [Quick checklist](#quick-checklist) | — |

---

## SECTION 0 — Build and Setup

### 0.1 Clean the old build

```bash
$ make clean
```

**What it does:** deletes everything produced by a previous build — the
`myshell` executable and all seventeen `.o` object files. This guarantees the
next build compiles every file from scratch, so no stale object file can hide a
compile error you introduced.

### 0.2 Compile

```bash
$ make
```

**What it does:** compiles each `.cpp` file into a `.o` object file, then links
them all together into the executable `myshell`.

**Expect exactly 17 compile lines then one link line, with no warnings and no
errors:**

```
g++ -Wall -Wextra -std=c++11 -g -c main.cpp -o main.o
g++ -Wall -Wextra -std=c++11 -g -c parser.cpp -o parser.o
... (15 more) ...
g++ -Wall -Wextra -std=c++11 -g -o myshell main.o parser.o ... -lreadline
```

`-Wall -Wextra` turn on the full warning set, so a clean run here is a real
result, not silence from a compiler that was not looking.

If it says `readline/readline.h: No such file`, install the library first:

```bash
$ sudo apt install libreadline-dev
```

### 0.3 Make a clean playground

> **Do NOT test inside your source folder** — you will fill it with junk files
> that then have to be cleaned out before submission.

```bash
$ mkdir -p /tmp/guide && cd /tmp/guide
```

**What it does:** two commands joined by `&&`.

- `mkdir -p /tmp/guide` creates the directory. The `-p` flag means *create
  parent directories as needed, and do not complain if it already exists*, so
  this is safe to re-run.
- `cd /tmp/guide` moves your terminal into it.

Your terminal moves from `~/PLACEMENT/MY_PROJECTS/POSIX_SHELL_IMPLEMENTATION`
to `/tmp/guide`.

### 0.4 Copy the executable across

```bash
$ cp /home/<you>/PLACEMENT/MY_PROJECTS/POSIX_SHELL_IMPLEMENTATION/myshell /tmp/guide/
```

**What it does:** copies the compiled binary into the playground.

> **Use the full source path, not `.`** — running `cp .../myshell .` while you
> are *already* in the source folder makes the source and destination the same
> file, and `cp` refuses.

### 0.5 Build the test fixtures

```bash
$ mkdir -p testdir/inner
```

Creates a nested structure `/tmp/guide/testdir/inner/`, used later to test `cd`,
`ls`, and above all **recursive** `search`.

```bash
$ echo "hello world" > file1.txt
```

`echo "hello world"` would normally print to the screen; the `>` sends that
output into `file1.txt` instead. The file now contains `hello world`.

```bash
$ printf 'banana\napple\ncherry\n' > fruits.txt
```

`printf` prints exactly what you give it, and `\n` means newline. So
`fruits.txt` holds three lines: banana, apple, cherry. They are deliberately
**out of alphabetical order** so that `sort` has visible work to do.

```bash
$ printf 'l1\nl2\nl3\nl4\nl5\n' > lines.txt
```

Five numbered lines, used for pipelines like `cat lines.txt | head -3 | tail -2`
where you need to know exactly which lines should survive.

```bash
$ touch alpha.txt alnum.txt
```

`touch` creates empty files. These two are named to **share the prefix `al`** —
that is the whole point, because Section 11 uses them to test TAB completion
with one match versus several.

```bash
$ echo hidden > .hidden.txt
```

The leading `.` makes it a hidden file on Linux. Section 3 uses it to prove that
plain `ls` hides it while `ls -a` shows it.

### 0.6 Clear the old history

```bash
$ rm -f ~/.myshell_history
```

`rm` removes; `-f` forces it not to complain if the file is absent; `~` is your
real home directory. This clears any history from earlier runs so the counts in
Section 12 start from a known state.

### 0.7 Start the shell

```bash
$ ./myshell
```

The `./` means *run the executable in this directory* — without it, bash would
search `$PATH` and not find it. The prompt appears:

```
<yourname@yourmachine:~>
```

The folder you started from becomes the shell's **home** and is shown as `~`.
To leave at any time: type `exit` or press **Ctrl-D**.

---

## SECTION 1 — Prompt

**Requirement 1**

The prompt must be `<username@systemname:directory>` with your **real**
username and machine name — both are read at runtime, nothing is hard-coded.

```
> pwd
-> /tmp/guide
```

Confirms the shell started where you think it did.

```
> cd testdir
-> prompt becomes:   <name@machine:~/testdir>
```

Moving down one level. The `~` stays because you are still inside the home.

```
> cd ..
-> prompt becomes:   <name@machine:~>
```

`..` means the parent directory, taking you back up.

```
> cd /etc
-> prompt becomes:   <name@machine:/etc>
```

An **absolute path**, outside the shell's home. There is no `~` here — the full
path is shown, which is exactly the point of this test.

```
> cd ~
-> back to  <name@machine:~>
```

✅ **Check:** `~` appears only for the start folder and things inside it. If
`/etc` also showed a `~`, the prefix test in the prompt code is wrong.

---

## SECTION 2 — cd, echo, pwd

**Requirement 2**

### 2.1 pwd and echo

```
> pwd
-> /tmp/guide

> echo Hello World
-> Hello World
```

Tests the built-in `echo` — not the system one.

```
> echo "abc  spaced   out"
-> abc  spaced   out
```

The multiple spaces **inside the quotes are kept**. This tests that the parser
treats a quoted string as one token instead of splitting on whitespace.

```
> echo
->
   (one empty line)
```

`echo` with no arguments still prints its newline.

### 2.2 All the cd forms

```
> cd testdir
> pwd
-> /tmp/guide/testdir

> cd ..
> pwd
-> /tmp/guide

> cd -
-> /tmp/guide/testdir
```

`cd -` returns to the **previous** directory and prints the path it moved to,
the same as bash. The shell tracks this itself, because the assignment forbids
using the `OLDPWD` environment variable.

```
> cd ~
> pwd
-> /tmp/guide

> cd .
-> nothing printed, you stay where you are
```

`.` means the current directory, so this is a legal no-op.

```
> cd
-> goes to the home of the shell (no argument = home)
```

### 2.3 cd errors

```
> cd a b
-> Invalid arguments
```

Two arguments after `cd` is an error, and the assignment specifies this exact
message.

```
> cd /nosuchdir
-> cd failed: No such file or directory
```

The message comes from `perror`, which turns the system's `errno` into readable
text.

---

## SECTION 3 — ls

**Requirement 3**

### 3.1 Plain and `-a`

```
> ls
-> alnum.txt
   alpha.txt
   file1.txt
   fruits.txt
   lines.txt
   myshell
   testdir
```

Note `.hidden.txt` is **absent**, and the names are sorted alphabetically —
`readdir` returns them in arbitrary order, so sorting is your code's doing.

```
> ls -a
-> .
   ..
   .hidden.txt
   alnum.txt
   ... (the rest)
```

`-a` means *all*, so the hidden entries now appear, including `.` and `..`.

### 3.2 Long format

```
> ls -l
-> total 1244
   -rw-rw-r-- 1 name name       0 Aug 27 22:19 alnum.txt
   ...
```

Each line shows permissions, link count, owner, group, size, modification time
and name — the same seven fields as the real `ls -l`.

### 3.3 ⭐ Compare against the real `ls -l`

This is the strongest single test of your `ls`.

```bash
$ exit                          # leave your shell first
$ ls -l > /tmp/real_ls.txt      # the REAL ls output
$ ./myshell
> ls -l > /tmp/mine_ls.txt      # YOUR ls output
> exit
$ diff /tmp/real_ls.txt /tmp/mine_ls.txt
```

**What `diff` does:** compares two files line by line and prints the
differences. **No output means the files are identical** — which is the result
you want.

✅ **Verified:** produces no output. The two are byte-identical, including the
`total` line and the column alignment.

### 3.4 Paths and flag combinations

```
> ls .
> ls ..
> ls ~
> ls testdir
> ls -l testdir
> ls -la
> ls -al
> ls -a -l
> ls -l testdir -a .
```

All of these must work. The last one mixes flags and folders **in any order**,
which the assignment explicitly requires.

### 3.5 ls errors

```
> ls nonexistent
-> ls: It cannot access nonexistent: No such file or directory

> ls -z
-> ls: It is invalid option -- 'z'
```

---

## SECTION 4 — System commands, foreground and background

**Requirement 4**

### 4.1 Foreground

```
> date
-> Wed Aug 27 22:31:07 IST 2025
```

`date` is an external Linux program, so this proves `fork` + `execvp` work.

```
> sleep 3
-> (the prompt comes back only after 3 seconds)
```

This is the definition of a **foreground** process: the shell waits for it.

```
> whoami
-> your username
```

### 4.2 Running your shell inside itself

```
> ./myshell
-> a second prompt appears
> exit
-> you are back in the first shell
```

The assignment requires this to work — your shell must be able to run another
copy of itself as an ordinary program.

### 4.3 Background with `&`

```
> sleep 30 &
-> 71301
```

The `&` means *do not wait*. The prompt returns **immediately** and the PID of
the new process is printed, as the assignment requires.

```
> sleep 40 &
-> 71302
```

A second one, to prove multiple background jobs coexist.

```
> jobs
-> [1] Running   sleep [71301]
   [2] Running   sleep [71302]
```

When each finishes, the shell prints a message like
`sleep with pid 71301 exited` — that comes from the SIGCHLD handler reaping it.

### 4.4 A command that does not exist

```
> nosuchcommand
-> nosuchcommand: No such file or directory
```

`execvp` failed in the child, and `perror` reported why.

---

## SECTION 5 — pinfo

**Requirement 5**

```
> pinfo
-> Process Status -- R
   memory -- 13234176 {Virtual Memory}
   Executable Path -- ~/myshell
```

With no argument, `pinfo` reports on **the shell itself**. The path is shown
with `~` because the binary sits in the shell's home folder.

```
> pinfo 1
-> Process Status -- S
   memory -- 24387584 {Virtual Memory}
   Executable Path -- /sbin/init
```

PID 1 is `init`/`systemd`, owned by root. This is a good test because your code
must fall back gracefully when it cannot read another user's process fully.

**Status letters:** `R` running, `S` sleeping, `Z` zombie, `T` stopped. A `+`
is appended when the process is in the **foreground**.

```
> pinfo abc
-> pinfo: PID is invalid

> pinfo 999999
-> pinfo: process with pid 999999 not found
```

The first is rejected because it is not a number; the second because no such
process exists.

---

## SECTION 6 — search

**Requirement 6**

```
> search file1.txt
-> True

> search testdir
-> True
```

The second one matters: `search` must find **directories**, not just files.

```
> search inner
-> True
```

`inner` lives at `testdir/inner`, one level down — so this proves the search
really is **recursive** rather than only looking in the current folder.

```
> search nothere
-> False
```

---

## SECTION 7 — I/O redirection

**Requirement 7**

### 7.1 Output with `>`

```
> echo hello > out.txt
-> (nothing on the screen)
> cat out.txt
-> hello
```

`>` sends standard output into a file instead of the terminal, so the absence of
screen output *is* the pass condition.

### 7.2 Append with `>>`

```
> echo appended >> out.txt
> cat out.txt
-> hello
   appended
```

`>>` adds to the end and keeps what was already there.

### 7.3 Overwrite with `>`

```
> echo fresh > out.txt
> cat out.txt
-> fresh
```

`>` truncates the file first, so the old contents are gone. This is the
difference between `>` and `>>`.

### 7.4 Input with `<`

```
> cat < fruits.txt
-> banana
   apple
   cherry
```

`cat fruits.txt` passes a *filename argument*; `cat < fruits.txt` instead
connects the file to the program's **standard input**. The output looks the
same, but the mechanism being tested is different.

### 7.5 Input and output together

```
> sort < fruits.txt > sorted.txt
> cat sorted.txt
-> apple
   banana
   cherry
```

Data flows `fruits.txt -> sort -> sorted.txt`, with nothing on screen.

### 7.6 ⭐ Redirection on your own built-ins

```
> ls > listing.txt
> cat listing.txt
-> (the file listing)

> echo hello > e.txt
> cat e.txt
-> hello
```

This is the important one. Built-ins run **inside** the shell process, so the
shell has to save its own stdout, point it at the file, run the command, flush,
and put stdout back. If the flush is missing you get an empty file and the text
on screen instead.

### 7.7 File permissions must be 0644

```bash
$ exit
$ stat -c '%a %n' out.txt
-> 644 out.txt
```

`stat -c '%a %n'` asks Linux for the permission bits and the filename. `644`
means owner read+write, group read, others read — exactly what the assignment
mandates.

### 7.8 Redirection errors

```
> cat < missing.txt
-> input file missing.txt: No such file or directory

> echo x >
-> Syntax error: file name is missing after '>'
```

The missing input file is reported by the shell **before** any process is
created.

---

## SECTION 8 — Pipelines

**Requirement 8**

A pipe `|` connects the output of the command on its left to the input of the
command on its right.

### 8.1 Two commands

```
> cat fruits.txt | wc -l
-> 3
```

`cat` emits three lines, `wc -l` counts them.

```
> cat fruits.txt | sort
-> apple
   banana
   cherry

> ls | grep txt
-> alnum.txt
   alpha.txt
   file1.txt
   fruits.txt
   lines.txt
```

`grep txt` keeps only the names containing `txt`.

```
> echo hello | wc -c
-> 6
```

✅ **Check:** `6` is the number that matters — five letters plus the newline. A
larger number means `echo` is emitting more than it should inside a pipeline.

### 8.2 Three and four commands

```
> ls | grep txt | wc -l
-> 5

> cat lines.txt | head -3 | tail -2
-> l2
   l3
```

`head -3` keeps `l1 l2 l3`; `tail -2` keeps the last two of those.

```
> cat /etc/passwd | cut -d: -f1 | sort | head -3
-> _apt
   avahi
   backup
```

Four stages: read the file, cut field 1 using `:` as the separator, sort, take
the first three. Names differ per machine.

### 8.3 ⭐ The hang test (very important)

```
> cat /etc/passwd | head -2
-> root:x:0:0:root:/root:/bin/bash
   daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
```

✅ **Check:** the prompt must come back **at once**. `head` stops after two
lines while `cat` is still writing. A reader only sees end-of-file when *every*
write end of the pipe is closed — so if the parent shell left any pipe
descriptor open, this hangs forever and you would have to press Ctrl-C. This
single test catches the classic pipeline bug.

---

## SECTION 9 — Redirection inside pipelines

**Requirement 9**

```
> ls | grep txt > txtfiles.txt
> cat txtfiles.txt
-> alnum.txt
   alpha.txt
   file1.txt
   fruits.txt
   lines.txt
```

The output of the **last** stage goes to the file, so nothing appears on screen.

```
> cat < fruits.txt | sort > s2.txt
> cat s2.txt
-> apple
   banana
   cherry
```

Input redirection at the start, a pipe in the middle, output redirection at the
end — `fruits.txt -> cat -> sort -> s2.txt`.

```
> ls | grep txt > out.txt ; cat < fruits.txt | wc -l > lines2.txt
```

Two independent pipelines separated by `;`, run one after the other. This is
the exact example given in the assignment.

---

## SECTION 10 — Signals **[MANUAL]**

**Requirement 10**

> These need a real terminal — piping input will not reproduce them.

### 10.1 Ctrl-C on a running command

```
> sleep 50
   (press Ctrl-C)
-> ^C
> echo still alive
-> still alive
```

✅ **Check:** the `sleep` dies but **the shell survives**. The shell catches
SIGINT and forwards it to the foreground child only.

### 10.2 Ctrl-C with nothing running

```
> (press Ctrl-C at an empty prompt)
-> nothing happens, the shell stays
```

The assignment requires exactly this: no foreground process means no effect.

### 10.3 Ctrl-Z

```
> sleep 50
   (press Ctrl-Z)
-> sleep with pid 71516 is stopped and sent to background
> jobs
-> [1] Stopped   sleep [71516]
```

The job is stopped, not killed, and moves into the job list. To clean it up
after leaving the shell:

```bash
$ kill 71516        # use the PID your shell printed
```

### 10.4 Ctrl-D

```
> (press Ctrl-D at an empty prompt)
-> the shell exits, you are back in bash
```

Ctrl-D is **not a signal** — it is end-of-file on the terminal. Your terminal
window itself must stay open.

---

## SECTION 11 — Autocomplete **[MANUAL]**

**Requirement 11**

### 11.1 One matching file

```
> cat alp<TAB>
-> cat alpha.txt
```

Only `alpha.txt` starts with `alp`, so it completes fully.

### 11.2 Several matching files

```
> cat al<TAB>
-> nothing completes yet (both alpha.txt and alnum.txt match)
> (press TAB again)
-> alnum.txt  alpha.txt
```

With more than one candidate, readline shows the list instead of guessing.

### 11.3 Command completion

```
> ech<TAB>
-> echo

> ca<TAB><TAB>
-> cat  cal  capsh  ...
```

The second one proves commands come from a real scan of `$PATH`, not a small
hard-coded list.

---

## SECTION 12 — history

**Requirement 12**

```
> ls
> cd
> ls
> history
-> ls
   cd
   ls
   history
```

The `history` command itself appears, matching the assignment's example.

```
> history 3
-> (only the last 3 entries)
```

### 12.1 It survives a restart

```
> exit
$ ./myshell
> history
-> the old commands are still there
```

They are stored in `~/.myshell_history`, written after **every** command — so
history survives even a crash, not just a clean exit.

### 12.2 The 20-command limit

```
> echo 1
> echo 2
   ... up to ...
> echo 25
> history 20
-> only the most recent 20 are kept
```

✅ **Verified:** 25 commands entered leaves exactly 20 stored, oldest evicted
first. Plain `history` shows 10 by default.

### 12.3 Arrow keys **[MANUAL]**

Press **↑** to walk back through previous commands and **↓** to come forward.
A recalled command can be edited before you press Enter.

---

## SECTION 13 — Mixed and tricky cases

```
> ls -l ; pwd ; echo three
```

Three commands separated by `;`, executed left to right.

```
>     ls    ;   pwd
```

Random spaces and tabs must be ignored — the assignment calls this out
specifically.

```
> echo     spaced      out
-> spaced out
```

Unquoted runs of spaces collapse, because the tokenizer drops empty tokens.
(Compare with the **quoted** version in Section 2.1, where they are kept.)

```
> echo hi>nospace.txt
> cat nospace.txt
-> hi
```

No spaces around `>`. The parser must still split this into three tokens.

```
> ls|wc -l
-> 8
```

Same idea for `|`.

```
> sleep 5 | wc &
```

A backgrounded pipeline. **Known limitation:** the `&` is accepted but the
pipeline still waits for its stages — measured at the full duration. The
assignment never combines `&` with `|`.

```
> cd testdir | wc -l
> pwd
-> /tmp/guide
```

✅ **Check:** the directory did **not** change. Every pipeline stage runs in a
forked child, so the `cd` happened in a process that immediately exited. Bash
behaves identically — this is correct, not a bug.

```
>
> (just spaces)
```

Empty and whitespace-only lines are ignored, and are not added to history.

---

## SECTION 14 — Crash test

Paste these in one at a time. The **exact** error text matters less than the
one rule that does matter:

> **The shell must never crash.** A `Segmentation fault` here is a grading
> failure.

```
> |
> |||
> > file
> <
> ;;;
> cd cd cd cd
> ls -----
> pinfo -5
> history -3
> search
> echo | | echo
> nosuchcmd | nosuchcmd2
```

✅ **Verified** — every one produces a clean error and the shell stays alive:

```
cd cd cd cd            -> Invalid arguments
ls -----               -> ls: It is invalid option -- '-'
pinfo -5               -> pinfo: PID is invalid
history -3             -> history: count must be positive
search                 -> Usage: search <filename>
echo | | echo          -> Syntax error: command should not be empty before/after '|'
nosuchcmd | nosuchcmd2 -> nosuchcmd: No such file or directory
```

---

## SECTION 15 — Rules check

Run these in your **normal bash terminal**, from the source folder.

### 15.1 Forbidden functions are not used

```bash
$ grep -n "popen\|pclose\|system(" *.cpp *.h
```

`grep -n` prints matching lines with line numbers. The `\|` means *or*.

✅ **Expected:** no output, apart from your own function named
`systemCommand` — which is your code, not the banned C `system()` call.

```bash
$ grep -n "filesystem" *.cpp *.h      # C++ filesystem library is banned
$ grep -n "ncurses" *.cpp *.h         # ncurses is banned
$ grep -n "OLDPWD\|getenv(\"PWD" *.cpp *.h    # these env vars are banned
```

All three should print nothing.

### 15.2 strtok **is** used — the assignment requires it

```bash
$ grep -n "strtok" parser.cpp
```

Should show it inside `SplitBySemicolon`.

### 15.3 execvp is used only for external commands

```bash
$ grep -ln "execvp" *.cpp
-> pipe.cpp
   system.cpp
```

`-l` prints only the filenames. Exactly these two are expected — `execvp` must
**not** appear in `cd.cpp`, `ls.cpp`, `echo.cpp` or `pwd.cpp`, because the
assignment forbids implementing your own built-ins with it.

### 15.4 No compiler warnings

```bash
$ make clean && make 2>&1 | grep -i warning
```

`2>&1` merges error output into normal output so `grep` can see it.

✅ **Expected:** no output at all.

---

## SECTION 16 — Before you submit

```bash
$ cd <source folder>
$ make clean
$ rm -f AOS_Assignment_2_M25_V1.pdf
$ ls
```

`make clean` removes the binary and the `.o` files — never submit build
artifacts. The `rm` clears the assignment PDF out of your submission folder.

You should be left with **only**:

```
README.md   TESTING_GUIDE.md   makefile   functions.h   and the 17 .cpp files
```

Then zip it with your roll number:

```bash
$ cd ..
$ zip -r <rollnumber>_Assignment2.zip <source folder>
```

**Last check** — unpack it somewhere else and confirm it still builds:

```bash
$ cd /tmp && unzip <rollnumber>_Assignment2.zip -d ziptest
$ cd ziptest/* && make
-> compiles with no errors
```

---

## The tests that matter most

If you are short on time, these carry the most marks:

| Test | What it proves |
|---|---|
| `diff` of `ls -l` vs the real `ls -l` | `ls` is correct down to the byte |
| `sleep 30 &` | Background execution, PID printed |
| `jobs` | Job management |
| `pinfo` and `pinfo 1` | `/proc` parsing without `popen` |
| `search inner` | Search really is recursive |
| `echo hello > out.txt` | Redirection works on **built-ins** |
| `sort < fruits.txt > sorted.txt` | Input and output redirection together |
| `cat fruits.txt \| wc -l` | Basic pipe |
| `cat /etc/passwd \| head -2` | Pipe descriptors closed — no hang |
| `ls \| grep txt > out.txt` | Redirection inside a pipeline |
| Ctrl-C / Ctrl-Z / Ctrl-D | Signal handling, shell survives |
| TAB | Autocomplete for files and PATH commands |
| `history` after restart | Persistence across sessions |
| `;;;`, `\|`, `<` and friends | No segmentation fault |
| `grep` checks in Section 15 | Assignment rule compliance |

---

## Quick checklist

- [ ] **0.** `make clean && make` — 17 compiles, 1 link, no warnings
- [ ] **1.** Prompt shows real `username@hostname` and `~` for the home
- [ ] **2.** `cd` (all forms), `echo`, `pwd`
- [ ] **3.** `ls` with `-a` `-l`, any order of flags and folders, `diff` is empty
- [ ] **4.** Foreground waits, background prints the PID, many jobs together
- [ ] **5.** `pinfo` for the shell and for any PID, `pinfo 1` shows a real path
- [ ] **6.** `search` gives True / False, finds folders and nested names
- [ ] **7.** Redirection `<` `>` `>>`, works on built-ins also, file is 0644
- [ ] **8.** Pipes of any length, `cat | head` does not hang
- [ ] **9.** Redirection inside pipes
- [ ] **10.** Ctrl-C, Ctrl-Z, Ctrl-D (shell survives the first two)
- [ ] **11.** TAB completion for files and for commands from PATH
- [ ] **12.** `history`, `history <n>`, survives a restart, arrow keys
- [ ] **13.** Semicolons, random spaces, symbols without spaces
- [ ] **14.** No segmentation fault on any rubbish input
- [ ] **15.** No `popen`/`system`/`filesystem`/`ncurses`, `strtok` is used
- [ ] **16.** `make clean` done before zipping
