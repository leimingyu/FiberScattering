#ifndef CMD_PARSER_H
#define CMD_PARSER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *sourcefile;

static const char *show_usage =
                                "Usage: ./fd [<options>]\n"
                                "Options:\n"
                                "\t--help\t\tShow this help message\n"
                                "\t--source\tSpecify the source file\n"
                                "\n";
static const char *err_note =
                "Please type 'fd --help' for a list of valid command-line options.\n";


static void need_argument(int argc, char **argv, int argi)
{
        if (argi == argc - 1)
		printf("option %s requires one argument.\n%s\n", argv[argi], err_note);
}

static void read_command_line(int *argc_ptr, char **argv)
{
	int argc = *argc_ptr;

	if (argc < 2 || argc > 3) {
		printf("%s", show_usage);	
		exit (EXIT_FAILURE);
	}

	int argi;
	for (argi = 1; argi < argc; argi++)
	{
		if (!strcmp(argv[argi], "--help")){
			printf("%s", show_usage);	
		}
		else if (!strcmp(argv[argi], "--source")) {
			need_argument(argc, argv, argi);
			sourcefile = argv[++argi];      
		}else{
			printf("%s is not a valid command-line option.\n%s\n", argv[argi], err_note); 
		}
	}
}



#endif
