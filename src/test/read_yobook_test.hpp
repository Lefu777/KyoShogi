#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include "cshogi.h"
#include "types.hpp"
#include "util.hpp"

void print_yo_book_test0(const std::vector<yo_book_t>& yo_books) {
	for (const auto& yo_book : yo_books) {
		std::cout << "[" << yo_book.sfen << "][" << yo_book.ply << "]" << std::endl;
		for (const auto& yo_book_move : yo_book.moves) {
			std::cout << "    "
				<< "[" << __move_to_usi(yo_book_move.move1) << "]"
				<< "[" << __move_to_usi(yo_book_move.move2) << "]"
				<< "[" << yo_book_move.eval << "]"
				<< "[" << yo_book_move.depth << "]"
				<< "[" << yo_book_move.select << "]"
				<< std::endl;
		}
	}
}

void read_yo_book_test0(const std::string& filename) {
	std::ifstream ifs(filename);
	std::string line;    // buffer
	__Board board;

	std::vector<yo_book_t> yo_books;
	while (std::getline(ifs, line)) {
		auto&& tokens = tokenize(line);
		if (line[0] == '#') {    // コメント
			std::cout << "comment: " << line << std::endl;
		}
		else if (tokens[0] == "sfen") {    // 局面
			// 0    1                                                                    2 3       4
			// sfen +B1g5l/R1g1k1gs1/2pp1p+Pp1/4pb1np/P1S6/L1PP4P/1P2PP+pS1/1+p1K1G3/7RL w SNP2nlp 88
			std::string&& sfen_without_ply = join(" ", tokens[1], tokens[2], tokens[3]);
			std::string&& sfen = join(" ", sfen_without_ply, tokens[4]);
			board.set(sfen);
			int ply_tmp = std::atoi(tokens[4].c_str());
			yo_books.emplace_back(sfen_without_ply, ply_tmp);

			//std::cout << "sfen: ["<< sfen_without_ply << "], ply = [" << ply_tmp << "]" << std::endl;
		}
		else {    // 指し手
			auto& crrt_book = yo_books.back();
			int move1 = board.move_from_usi(tokens[0]);
			board.push(move1);
			int move2 = board.move_from_usi(tokens[1]);
			board.pop();
			crrt_book.moves.emplace_back(
				move1,
				move2,
				std::atoi(tokens[2].c_str()),
				std::atoi(tokens[3].c_str()),
				std::atoi(tokens[4].c_str())
			);
		}
	}

	ifs.close();

	print_yo_book_test0(yo_books);
}

void print_yo_book_test1(const std::unordered_map<std::string, yo_book_value_t>& yo_books) {
	for (const auto& kv : yo_books) {
		const auto& sfen = kv.first;
		const auto& yo_book_value = kv.second;
		std::cout << "[" << sfen << "][" << yo_book_value.ply << "]" << std::endl;
		for (const auto& yo_book_move : yo_book_value.moves) {
			std::cout << "    "
				<< "[" << __move_to_usi(yo_book_move.move1) << "]"
				<< "[" << __move_to_usi(yo_book_move.move2) << "]"
				<< "[" << yo_book_move.eval << "]"
				<< "[" << yo_book_move.depth << "]"
				<< "[" << yo_book_move.select << "]"
				<< std::endl;
		}
	}
}

void read_yo_book_test1(const std::string& filename) {
	std::ifstream ifs(filename);
	std::string line;    // buffer
	__Board board;

	std::unordered_map<std::string, yo_book_value_t> yo_books;
	std::string crrt_key;

	while (std::getline(ifs, line)) {
		auto&& tokens = tokenize(line);
		if (line[0] == '#') {    // コメント
			std::cout << "comment: " << line << std::endl;
		}
		else if (tokens[0] == "sfen") {    // 局面
			// 0    1                                                                    2 3       4
			// sfen +B1g5l/R1g1k1gs1/2pp1p+Pp1/4pb1np/P1S6/L1PP4P/1P2PP+pS1/1+p1K1G3/7RL w SNP2nlp 88
			std::string&& sfen_without_ply = join(" ", tokens[1], tokens[2], tokens[3]);
			std::string&& sfen = join(" ", sfen_without_ply, tokens[4]);
			board.set(sfen);
			int ply_tmp = std::atoi(tokens[4].c_str());
			auto&& status = yo_books.emplace(sfen_without_ply, yo_book_value_t{ ply_tmp });
			if (!status.second) {
				std::cout << "Error: dupilication sfen was found, sfen = " << sfen << std::endl;
				exit(1);
			}

			crrt_key = std::move(sfen_without_ply);

			//std::cout << "sfen: ["<< sfen_without_ply << "], ply = [" << ply_tmp << "]" << std::endl;
		}
		else {    // 指し手
			auto& crrt_book = yo_books[crrt_key];
			int move1 = board.move_from_usi(tokens[0]);
			board.push(move1);
			int move2 = board.move_from_usi(tokens[1]);
			board.pop();
			crrt_book.moves.emplace_back(
				move1,
				move2,
				std::atoi(tokens[2].c_str()),
				std::atoi(tokens[3].c_str()),
				std::atoi(tokens[4].c_str())
			);
		}
	}

	ifs.close();

	print_yo_book_test1(yo_books);
}