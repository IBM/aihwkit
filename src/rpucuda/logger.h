#pragma once

#include <stdio.h>
#include <ctime>


class Logger
{
private:
	static const char* logger_path;
	static FILE* file;

public:

	static void EnableFileOutput(const char* new_filepath)
	{
		logger_path = new_filepath;
		enable_file_output();
	}

	static void CloseFileOutput()
	{
		free_file();
	}

	template<typename... Args>
	static void log(const char* message, Args... args)
	{
		std::time_t current_time = std::time(0);
		std::tm* timestamp = std::localtime(&current_time);
		char buffer[80];
		strftime(buffer, 80, "%c", timestamp);

		printf("%s\t", buffer);
		printf(message, args...);
		printf("\n");

		if (file)
		{
			fprintf(file, "%s\t", buffer);
			fprintf(file, message, args...);
			fprintf(file, "\n");
		}
	}

private:

	static void enable_file_output()
	{
		if (file != 0)
		{
			fclose(file);
		}

		file = fopen(logger_path, "a");

		if (file == 0)
		{
			printf("Logger: Failed to open file at %s", logger_path);
		}
	}

	static void free_file()
	{
		fclose(file);
		file = fopen(logger_path, "a");
	}
};

const char* Logger::logger_path = "log.txt";
FILE* Logger::file = fopen(logger_path, "a");