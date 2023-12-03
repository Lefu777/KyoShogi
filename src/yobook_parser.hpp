#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <iterator>
#include "cshogi.h"
#include "types.hpp"
#include "util.hpp"
#include "config.hpp"

class YoBookParser {
private:
	std::unordered_map<std::string, yo_book_value_t> _books;

public:
	YoBookParser() {

	}

	size_t size() const {
		return _books.size();
	}

	// TOOD
	//     : 私、手番無しとか書いてるけど、手数の間違いだよな？もう記憶ねぇわ。
	// 指定された局面に対応する情報を探す。
	// @arg _sfen
	//     : ex1> tokens.size() == 3
	//        0                                                               1 2
	//       "+B+Rsg2snl/4k1gp1/p2ppp2p/2p4b1/9/6P2/P2PPP2P/2G1K1S2/+r1S2G1NL w NL2Pnl3p"
	//     : ex2> tokens.size() == 4
	//        0                                                               1 2        3
	//       "+B+Rsg2snl/4k1gp1/p2ppp2p/2p4b1/9/6P2/P2PPP2P/2G1K1S2/+r1S2G1NL w NL2Pnl3p 36"
	auto find(const std::string& _sfen) {
		auto&& tokens = tokenize(_sfen);

		if (tokens.size() == 3) {    // 手数無し
			auto&& sfen = _sfen;
			// iter.first : key
			// iter.second : value
			auto&& iter = _books.find(sfen);
			if (iter == _books.end()) {
				return std::make_pair(false, iter);
			}
			else {
				return std::make_pair(true, iter);
			}
		}
		else if (tokens.size() == 4) {    // 手数付き
			// 手番を削る
			auto&& sfen = join(" ", tokens[0], tokens[1], tokens[2]);

			auto&& iter = _books.find(sfen);
			if (iter == _books.end()) {
				return std::make_pair(false, iter);
			}
			else {
				return std::make_pair(true, iter);
			}
		}
		else {
			throw std::runtime_error("Error: YoBookParser::find() got unexpected tokens.size()");
		}
	}

	bool parse(const std::string& filename) {
		_books.clear();

		std::ifstream ifs(filename);

		if (!ifs) {
			std::cout << "Error: failed to open " << filename << std::endl;
			return false;
		}

		std::string line;    // buffer
		std::string crrt_key;
		__Board board;

		while (std::getline(ifs, line)) {
			auto&& tokens = tokenize(line);
			if (line[0] == '#') {    // コメント
				//std::cout << "comment: " << line << std::endl;
			}
			else if (tokens[0] == "sfen") {    // 局面
				// 0    1                                                                    2 3       4
				// sfen +B1g5l/R1g1k1gs1/2pp1p+Pp1/4pb1np/P1S6/L1PP4P/1P2PP+pS1/1+p1K1G3/7RL w SNP2nlp 88
				std::string&& sfen_without_ply = join(" ", tokens[1], tokens[2], tokens[3]);
				std::string&& sfen = join(" ", sfen_without_ply, tokens[4]);
				board.set(sfen);
				int ply_tmp = std::atoi(tokens[4].c_str());
				auto&& status = _books.emplace(sfen_without_ply, yo_book_value_t{ ply_tmp });
				if (!status.second) {
					std::cout << "Error: dupilication sfen was found, sfen = " << sfen << std::endl;
					exit(1);
				}

				crrt_key = std::move(sfen_without_ply);

				//std::cout << "sfen: ["<< sfen_without_ply << "], ply = [" << ply_tmp << "]" << std::endl;
			}
			else {    // 指し手
				auto& crrt_book = _books[crrt_key];
				Move move1 = board.Move_from_usi(tokens[0]);
				board.push(move1);
				Move move2 = board.Move_from_usi(tokens[1]);    // NOTE: none ならちゃんとmoveNone() が格納されるはず。
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

#ifdef DEBUG_
		std::cout << "info string parse " << filename << " done" << std::endl;
		std::cout << "info string size() = [" << size() << "]" << std::endl;
		std::cout << "info string _books.size() = [" << _books.size() << "]" << std::endl;
#endif

		return true;
	}
};
