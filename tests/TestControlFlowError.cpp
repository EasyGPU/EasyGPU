/**
 * @file TestControlFlowError.cpp
 * @brief Error messages when called outside Kernel definition.
 */

#include <Flow/BreakFlow.h>
#include <Flow/ContinueFlow.h>
#include <Flow/DoWhileFlow.h>
#include <Flow/ForFlow.h>
#include <Flow/IfFlow.h>
#include <Flow/WhileFlow.h>

#include <iostream>
#include <string>

using namespace GPU::Flow;

static int	testsPassed = 0;
static int	testsTotal	= 0;

static bool checkErrorContains(const std::exception &e, const std::string &keyword) {
	std::string msg(e.what());
	return msg.find(keyword) != std::string::npos;
}

#define RUN_TEST(name, expr, expectedFunc)                                                                             \
	do {                                                                                                               \
		std::cout << "[Test " << #name << "] " << #expr << " ... " << std::flush;                                      \
		testsTotal++;                                                                                                  \
		bool caught = false;                                                                                           \
		try {                                                                                                          \
			expr;                                                                                                      \
		} catch (const std::exception &e) {                                                                            \
			caught = true;                                                                                             \
			if (checkErrorContains(e, expectedFunc)) {                                                                 \
				std::cout << "PASS" << std::endl;                                                                      \
				testsPassed++;                                                                                         \
			} else {                                                                                                   \
				std::cout << "FAIL (wrong message: " << e.what() << ")" << std::endl;                                  \
			}                                                                                                          \
		}                                                                                                              \
		if (!caught) {                                                                                                 \
			std::cout << "FAIL (no exception)" << std::endl;                                                           \
		}                                                                                                              \
	} while (0)

int main() {
	std::cout << "=== Control Flow Error Message Tests ===" << std::endl;

	// If called outside kernel should throw with "If" in message
	RUN_TEST(1, If(true, []() {}), "If");

	// For called outside kernel should throw with "For" in message
	RUN_TEST(2, For(0, 10, 1, [](auto &) {}), "For");

	// While called outside kernel should throw with "While" in message
	RUN_TEST(3, While(true, []() {}), "While");

	// DoWhile called outside kernel should throw with "DoWhile" in message
	RUN_TEST(4, DoWhile([]() {}, true), "DoWhile");

	// Break called outside kernel should throw with "Break" in message
	RUN_TEST(5, Break(), "Break");

	// Continue called outside kernel should throw with "Continue" in message
	RUN_TEST(6, Continue(), "Continue");

	// Verify the public-facing operation name remains present.
	RUN_TEST(7, If(true, []() {}), "If()");

	std::cout << "\n========================================" << std::endl;
	std::cout << "Test Results: " << testsPassed << "/" << testsTotal << " passed" << std::endl;
	std::cout << "========================================" << std::endl;

	return (testsPassed == testsTotal) ? 0 : 1;
}
