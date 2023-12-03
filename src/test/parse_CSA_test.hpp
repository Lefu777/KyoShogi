#pragma once

#include <iostream>

#include "parser.h"

void parse_csa_test0() {
	const std::string file_path = "wdoor+floodgate-300-10F+4t4t4t+AobaZero_w4153_n_p800+20221004193010.csa";

	//std::ifstream reading_file;
	//reading_file.open(file_path, std::ios::in);
	//std::string buf;
	//std::string file_str;
	//while (std::getline(reading_file, buf)) {
	//	file_str += buf + "\n";
	//}

	std::cout << "==================================================" << std::endl;
	//std::cout << file_str << std::endl;
	std::cout << "==================================================" << std::endl;

	parser::__Parser parser;
	int i = 0;

	std::cout << "info: parse_csa_str() start" << std::endl;
	//parser.parse_csa_str(file_str);
	parser.parse_csa_file(file_path);
	std::cout << "info: parse_csa_str() done" << std::endl;

	std::cout << "version = [" << parser.version << "]" << std::endl;
	std::cout << "sfen = [" << parser.sfen << "]" << std::endl;
	std::cout << "endgame = [" << parser.endgame << "]" << std::endl;
	std::cout << "comment = [" << parser.comment << "]" << std::endl;

	////enum GameResult : int8_t {
	////	Draw, BlackWin, WhiteWin, GameResultNum
	////};
	//// がset されている。
	//std::cout << "win = [" << parser.win << "]" << std::endl;

	//i = 0;
	//for (const auto& information : parser.informations) {
	//	std::cout << "informations[" << i << "]= [" << information << "]" << std::endl;
	//	++i;
	//}

	//i = 0;
	//for (const auto& name : parser.names) {
	//	std::cout << "names[" << i << "]= [" << name << "]" << std::endl;
	//	++i;
	//}

	//// 必要
	//i = 0;
	//for (const auto& rating : parser.ratings) {
	//	std::cout << "ratings[" << i << "]= [" << rating << "]" << std::endl;
	//	++i;
	//}

	// 必要
	// NOTE
	//     : 投了の直前までの指し手が格納されてる。
	//       まぁ、要するに"指し手"として有効な手のみが格納されている。
	i = 0;
	for (const auto& move : parser.moves) {
		std::cout << "moves[" << i << "]= [" << move << "]" << std::endl;
		++i;
	}
	std::cout << "parser.moves.size() = [" << parser.moves.size() << "]" << std::endl;

	//i = 0;
	//for (const auto& time : parser.times) {
	//	std::cout << "times[" << i << "]= [" << time << "]" << std::endl;
	//	++i;
	//}

	// 必要
	// TODO
	//     : 初めの32手いないで、評価値が1000を超えている物は省く
	//     : 結論が引き分けでないのに、評価値が終始0 のものは省く
	// NOTE
	//     : 先手有利なら+
	//       後手有利なら-
	i = 0;
	for (const auto& score : parser.scores) {
		std::cout << "scores[" << i << "]= [" << score << "]" << std::endl;
		++i;
	}

	//i = 0;
	//for (const auto& comment : parser.comments) {
	//	std::cout << "comments[" << i << "]= [" << comment << "]" << std::endl;
	//	++i;
	//}
}