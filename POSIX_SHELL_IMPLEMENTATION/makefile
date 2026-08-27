CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11 -g
LDFLAGS = -lreadline
TARGET = myshell

OBJS = main.o parser.o executor.o display.o cd.o echo.o pwd.o ls.o \
       pinfo.o search.o history.o jobs.o system.o pipe.o redirection.o \
       signal.o autocomplete.o

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

main.o: main.cpp functions.h
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

parser.o: parser.cpp functions.h
	$(CXX) $(CXXFLAGS) -c parser.cpp -o parser.o

executor.o: executor.cpp functions.h
	$(CXX) $(CXXFLAGS) -c executor.cpp -o executor.o

display.o: display.cpp functions.h
	$(CXX) $(CXXFLAGS) -c display.cpp -o display.o

cd.o: cd.cpp functions.h
	$(CXX) $(CXXFLAGS) -c cd.cpp -o cd.o

echo.o: echo.cpp functions.h
	$(CXX) $(CXXFLAGS) -c echo.cpp -o echo.o

pwd.o: pwd.cpp functions.h
	$(CXX) $(CXXFLAGS) -c pwd.cpp -o pwd.o

ls.o: ls.cpp functions.h
	$(CXX) $(CXXFLAGS) -c ls.cpp -o ls.o

pinfo.o: pinfo.cpp functions.h
	$(CXX) $(CXXFLAGS) -c pinfo.cpp -o pinfo.o

search.o: search.cpp functions.h
	$(CXX) $(CXXFLAGS) -c search.cpp -o search.o

history.o: history.cpp functions.h
	$(CXX) $(CXXFLAGS) -c history.cpp -o history.o

jobs.o: jobs.cpp functions.h
	$(CXX) $(CXXFLAGS) -c jobs.cpp -o jobs.o

system.o: system.cpp functions.h
	$(CXX) $(CXXFLAGS) -c system.cpp -o system.o

pipe.o: pipe.cpp functions.h
	$(CXX) $(CXXFLAGS) -c pipe.cpp -o pipe.o

redirection.o: redirection.cpp functions.h
	$(CXX) $(CXXFLAGS) -c redirection.cpp -o redirection.o

signal.o: signal.cpp functions.h
	$(CXX) $(CXXFLAGS) -c signal.cpp -o signal.o

autocomplete.o: autocomplete.cpp functions.h
	$(CXX) $(CXXFLAGS) -c autocomplete.cpp -o autocomplete.o

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: clean
